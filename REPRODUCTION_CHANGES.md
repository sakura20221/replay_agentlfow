# 当前复现协议与修改边界

本文只描述当前可运行协议。历史运行事故和已被后续核验推翻的判断不属于协议，
因此不再保留在这里。作者仓库固定在 `upstreams.lock.json` 的完整 commit，所有本地
修改由 `shims/` 中的安装器从零重建并校验。

## 1. 实验矩阵

主比较包含 7 个方法和 5 个数据集，共 35 个单次运行单元：AFlow、MaAS、DAAO、
G-Designer、CARD、FlowBank、MasRouter，分别在 MATH、AMC、MBPP、DROP、MMLU-Pro
上运行。

另有 G-Designer 和 CARD 的 `*_authordefault` 两行，每行 5 个数据集，只用于显示
作者默认 10 次迭代（40 道搜索题）的控制结果，不进入 7 方法主排名。因此代码的
完整静态矩阵是 9 行 x 5 列 = 45 格。

所有主方法在同一数据集上读取完全相同、顺序固定的搜索题和留出评测题。方法的
拓扑、角色、算子、损失、优化循环和作者超参数尽量保留；只有数据接入、任务域提示、
统一判分和运行可靠性被适配。

## 2. 固定数据

| 数据集 | 搜索 | 留出评测 | 来源 |
|---|---:|---:|---|
| MATH | 119 | 486 | AFlow/FlowBank 发布包的全量 Level-5 validate/test |
| AMC | 165 | 648 | FlowBank 发布包的 validate/test；删除两者内容重叠后评测 |
| MBPP | 256 | 500 | 官方 train 固定抽样 / 官方 test 全量（task 11-510） |
| DROP | 256 | 1000 | 官方 train / validation 分别固定抽样 |
| MMLU-Pro | 252 | 1120 | 官方 test 池按 14 学科分层切分，18/80 每学科 |

随机种子为 `20260821`。`shared/data/manifest.json` 记录来源、行数和每个文件的完整
SHA-256。预检同时验证：uid 唯一、搜索/评测内容零重叠、拼接文件严格等于
`search + eval`、MBPP 入口函数由测试正确推断、MMLU-Pro 搜索集为 14 x 18。

需要申报的边界：FlowBank 没有给 AMC 文件记录原始出处；MMLU-Pro 没有独立官方
训练集，因此搜索和评测从官方 test 池做了无内容重叠的分层切分。

## 3. 统一模型与生成协议

- 优化器和执行器都使用本地 `Qwen/Qwen3-8B`。
- `temperature=0`、`presence_penalty=1.0`、`enable_thinking=false`。
- 单次回复上限 8192 token；代理对外协议窗口为 32768，后端 vLLM 分配 40960。
- 回复撞满 8192 时，对完全相同的请求重新生成一次；不续写、不拼接两次回复。
- 调用方要求流式时，代理在后端非流式处理后返回一个合法 SSE 事件，兼容 MaAS。
- 只接受真实服务的模型别名，避免把多个虚假模型名静默映射到同一模型。
- 每次请求保存命名空间、输入、输出、token、截断、重试和错误信息。

每个作业写 `protocol.json`。当前指纹覆盖公共提示、完整判分源码、全部 shim/运行
适配源码、数据字节、采样参数、方法、数据集、repeat 和 run tag。收集器拒绝任何
不一致；续跑器也拒绝给旧搜索产物重写新协议标签。

## 4. 统一题面与判分

`shared/bench.py` 是唯一判分入口。每个方法仍可保留自己的内部提示和协作过程，
但接收相同的 canonical 题面，并在末尾收到与数据集一致的输出格式要求。

- MATH/AMC：AFlow 判分器为第一层；仅在其判错后尝试对称的 LaTeX 排版、变量赋值、
  单位装饰、无序解集和向量记法等价。原判正确的答案不会被降分。
- MBPP：执行官方测试；入口函数从测试调用推断；执行 `test_imports`/fixture；保留
  测试依赖的顶层类；仅剥除整段代码共有的外层缩进，内部相对缩进不变。
- DROP：使用官方 token F1，取多位标注者答案中的最大值，保留部分得分。
- MMLU-Pro：从最终回答抽取合法选项字母并做 exact match。

每个数据集的 gold answer 都会反向送入判分器。Linux/A800 必须全过；macOS 的
`libm` 浮点差异会使 MBPP 的两个任务在两个包装路径下共出现 4 次失败，不能据此
放宽官方测试。

## 5. 各方法适配

### MaAS 与 DAAO

保留作者控制器、动作采样、损失和训练/测试流程。新增五个共享数据配置，算子清单
从各自仓库真实模板读取，避免 DAAO 已删除算子与配置索引不一致。为每个数据集生成
对应任务域提示；MBPP 补齐公开测试查找和 task identity；字母选项任务限制
ScEnsemble 的候选标签空间；执行模型生成代码时加超时；结果保存完整题目和回复。

DAAO 的辅助 VAE 标记保持作者/用户确认的语义：DROP 只要 `F1 > 0` 就令
`is_solved=1`，部分 F1 本身仍作为统一 reward 保留。没有把它改成只有 `F1=1`
才算 solved。

### AFlow

保留作者“生成候选工作流 -> 验证 -> 迭代改写”的流程和默认失败重试预算。新增五个
共享 benchmark 配置及对应 workspace；依据相近的作者原生域继承算子集合。优化器
提示只补充算子构造函数的真实必需参数，避免候选因接口信息缺失全部运行失败。
MBPP 的 Test 算子按题目身份取本题官方测试。搜索结束后，按当前统一判分重算各轮
验证分，选择最高轮并在留出集执行。

### FlowBank

保留 DiverseFlow 候选搜索、CuraFlow portfolio 选择、QueryMatching selector 训练
和一次性留出测试。聚合以 uid 而不是题面为键，相同题面不同 task id 不会合并；
任何 workflow 漏题或 portfolio 文件缺失都会直接失败，不再用前几轮静默兜底。
MBPP Test 与 AFlow 使用同一 task-aware 公开测试查找。QueryMatching 使用统一的
本地 `all-MiniLM-L6-v2`；模型文件不进 Git，需单独下载。

### G-Designer 与 CARD

`run_shared.py` 分别从两个仓库自己的 `run_gsm8k.py` 定点派生，保留原 topology
policy-gradient 更新。按作者已有 math/code/MMLU runner 选择对应 agent 类、决策头、
角色数和角色配置。主行训练遍历完整搜索集；越过严格 item boundary 后关闭优化器，
只做留出评测；最后不足一个 batch 的题不会丢。每个配置写独立逐题结果文件，避免
时间戳同名混写。模型生成代码的执行路径加 30 秒保护，但不修正错误代码。

### MasRouter

`run_shared.py` 从作者 `run_math.py` 定点派生，保留 task loss、answer loss、VAE loss
和路由采样。MATH/AMC 映射 Math，MBPP 映射 Code，DROP/MMLU-Pro 映射作者用于问答的
Commonsense；只替换与答案形状冲突的角色措辞和 final-node 输出格式。训练和测试都
保留最后不足 16 的 batch。并发执行时 Graph 是每个 query 的局部对象，不写共享
`self.g`，所以不会跨题串图；逐题输入、回复、reward 和 uid 均落盘。

由于实验只提供一个 Qwen3-8B，作者的多 LLM 路由轴不能真实比较，配置被收缩为一个
如实命名的模型；任务类型、协作模式、agent 数和角色路由仍学习。这是协议限制，
不是运行 bug，结果中必须申报。

## 6. 明确没有改动的部分

- 没有改各论文方法的优化目标、reward 定义或核心搜索策略。
- 没有为某个方法单独增加搜索题、评测题或答案信息。
- 没有把错误代码、函数名不匹配或语法错误强行修成正确答案。
- 没有把 DROP 改成 exact match；官方 F1 和部分得分保持不变。
- 没有在留出集上选择工作流、轮次、portfolio 或 selector epoch。

运行可靠性修改包括：可追踪命名空间、逐题记录、稳定 uid、协议隔离、非 daemon
执行保护、代码超时、完整覆盖断言和错误即停。这些修改不替方法做决策，只防止静默
空输出、漏题、混写、挂死或错误收集。

## 7. 上游文件级核对范围

| 上游家族 | 安装器只允许的修改类别 |
|---|---|
| MaAS / DAAO | evaluator/config 注册、共享数据与提示、LLM 连接、判分转发、MBPP 测试身份、超时与记录 |
| AFlow | shared benchmark/workspace 注册、LLM 连接、接口说明、MBPP 测试身份、轮次保存与留出执行 |
| FlowBank | shared benchmark/workspace、uid 聚合、portfolio 完整性、MiniLM selector 输入、运行记录 |
| G-Designer / CARD | 从作者 runner 派生的共享入口、域配置、评测边界、结果文件、执行超时 |
| MasRouter | 从作者 runner 派生的共享入口、域角色/final prompt、单模型配置、局部 Graph 并发、逐题记录 |

安装器按 locked commit 的固定锚点修改；锚点数量不符、生成文件不能编译、重复安装不
幂等或关键断言不成立都会失败。`scripts/bootstrap_upstreams.py --check` 会逐仓库验证
commit 和全部安装器断言。

## 8. 运行门禁

`pipeline.py` 的顺序为：归档旧产物 -> 安装器 -> 静态预检 -> 45 格 6 题 smoke ->
live prompt/判分/收集审计 -> 再次归档 smoke -> 正式两波 sweep。AFlow 留出评测已在
sweep 内完成；FlowBank 的 portfolio/selector stages 2a-3e 在 search 后按数据集运行。

预检至少覆盖：15 个数据文件的字节哈希、45 格命令和边界、806 个 live prompt、
FlowBank uid/portfolio、MBPP task identity、gold replay、执行超时、代理协议和协议
隔离。任一项失败不得启动正式实验。
