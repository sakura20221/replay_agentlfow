# 复现修改清单

**生成时间**:2026-08-21 · **依据**:各仓库 `git diff --numstat` 与 `git status` 实测结果,非事后回忆

**当前进度**:7 个仓库已固定 commit;**7 / 7 个方法已全部接入共享层并通过双向判分测试**。四个家族 shim 全部完成:MaAS 系(MaAS + DAAO)、G-Designer 系(G-Designer + CARD)、MasRouter、AFlow。

---

## 0. 仓库与固定版本

| 方法 | 仓库 | commit | 体积 |
|---|---|---|---|
| AFlow | FoundationAgents/AFlow | `3f45721` | 4.4 M |
| MaAS | bingreeky/MaAS | `987f3c1` | 8.6 M |
| DAAO | AutoAgents-ai/DAAO | `5e260bb` | 15 M(稀疏检出) |
| G-Designer | yanweiyue/GDesigner | `a6efcfa` | 1.9 M |
| CARD | Warma10032/CARD | `d5d1f68` | 2.2 M |
| FlowBank | lingzhiyxp/FlowBank | `dde948d` | 45 M |
| MasRouter | yanweiyue/masrouter | `e005f76` | 6.2 M |

---

## 1. 全局协议(七个方法完全一致)

| 项 | 值 | 实现位置 |
|---|---|---|
| Executor 与 Optimizer | 同一个本地 `Qwen/Qwen3-8B` | 两个 vLLM 实例(GPU 1/7,各 32 GB) |
| temperature | 强制 0,调用方设的值被覆盖 | 代理 `_normalize` |
| thinking | 强制关闭,经 `chat_template_kwargs.enable_thinking=false` | 代理 `_normalize` |
| max_tokens | 统一 8192,**调用方设更大值时由代理压回并计数** | `shared/bench.py: MAX_TOKENS` + 代理 `_normalize` |
| 模型名 | 只接受 `Qwen/Qwen3-8B` / `qwen3-8b`,**其余一律拒绝** | 代理 `_normalize` |
| 上下文窗口 | 32768 | `launch_vllm.sh: MAX_MODEL_LEN` |
| 采样参数 | `top_p`/`top_k`/`presence_penalty`/`frequency_penalty` 一律剥离 | 代理 `_normalize` |
| 判分 | 单一权威,MATH/AMC/MBPP/DROP 直接调用 **AFlow 未修改的评测器** | `shared/bench.py` |
| 搜索 / 测试记账 | 按 URL namespace 分离(`train/*` 与 `test/*`) | 代理 |
| 诊断字段 | 截断率、thinking 泄漏、恢复次数、空回复、每 namespace token | 代理 `/stats` + `logs/api_calls.jsonl` |

### 1.1 代理引入的六个行为(七个方法一视同仁)

1. **截断续写**:`finish_reason == "length"` 时追加一次 ≤256 token 的短调用索取最终答案,拼接后返回,并记 `recovered`。
   *理由*:被截断回答判 0,而各方法截断率不同(拆解型 workflow 每节点输出短、单链 CoT 易撞上限),而方法间真实差距只有 1–4 分,截断噪声会盖过信号。
2. **上下文裁剪**:prompt + max_tokens 超窗时,依 vLLM 400 报错中的 token 数迭代下调 max_tokens(最多 8 轮,含严格递减保证)。
   *理由*:vLLM 硬报 400 不自动裁剪;多 agent 辩论累积历史最容易触发,不处理会系统性惩罚通信最丰富的方法。
   *已知边界*:prompt 本身 ≥ 25k token 时仍失败(见 §4)。
3. **listen backlog 512**:`ThreadingHTTPServer` 默认 5,并发突发时静默 RST 丢请求(实测 48 次丢 2 次)。
4. **流式请求以 SSE 原样应答**:代理需要完整响应体才能剥离 thinking、续写截断、记账,因此对上游一律用非流式;但若调用方请求 `stream=true`,回复会**重新包装成一个 SSE chunk** 返回,而不是把请求改成非流式。
   *理由*:早期版本直接把 `stream` 改写为 `false`。MaAS 的 `llm_config.stream` 默认为 `True`,它的 SSE 解析器在纯 JSON 响应体里找不到 `data:` 行,于是返回空字符串 —— 表现为 21 次 pydantic「缺字段」校验错误、11 次空输出、整批零梯度,而**任何一层都没有报错**。
5. **不认识的模型名直接拒绝**:早期版本无条件 `payload["model"] = UPSTREAM_MODEL`。
   *理由*:MasRouter 的 `llm_profile` 列了 5 个云模型(各自标注不同价格与 MMLU/HumanEval/MATH 分数),静默改写会让它的 LLM 路由头在「描述声称互不相同、实际全是同一个模型」的 5 个选项上学分布,每个请求都成功、数字都合理、日志里没有一处指出池子是假的。现在按 `requested_model` 计数并暴露在 `/stats`。
6. **max_tokens 上限**:超过 8192 的请求压回 8192,并在 `/stats.max_tokens_caps` 记下 `原值->8192`。
   *理由*:MasRouter 请求 81920,超过 `max_model_len` 会被 vLLM 直接 400 拒绝;同时统一预算才能让对比比的是工作流质量而不是 max_tokens 设置。这是**协议强制**,所以必须可见 —— 与第 5 条要避免的静默兜底是同一类教训。

### 1.2 判分层的信封解包(七个方法一视同仁)

各仓库的算子统一返回 `{"response": ...}` 信封,作者手写的种子工作流都会先解包再返回。**小模型优化器生成的工作流经常忘记解包**:FlowBank 的 round 2 三个步骤全部直传字典,`return final_answer` 返回的是字典,256 题全判 0 —— 得分 `0.0000`,在结果表里与「答全错」无法区分,而且不抛异常,所以重试预算永远不触发。

`shared/bench.py` 因此在评分入口做一次**机械解包**(取 `response`/`answer`/`output`/`solution` 中的字符串值),并计数 `unwrapped_envelope` / `unwrappable_dict`。

这**不是**作者的补救手段:DiverseFlow 原本会调 `llm_extract_answer` 用 LLM 从原始输出里抽答案,那同样能救回这一例,但会让 FlowBank 按比其余六个方法更宽松的尺度判分。解包只动容器、不动内容,对任何方法一视同仁。实测:同一份 round-2 图,修复前 `0.0000`,修复后 `0.6602`。

### 1.3 答案格式要求 + 格式示例(七个方法收到完全相同的文本)

**问题**:判分只认少数几种输出形态。实测 24 种真实(Qwen3-8B 跑出来的)输出形态,MATH/AMC 只有 3/6 判对、MMLU-Pro 4/6 —— 答对但写成 `Thus, 142.` 或 `Option J is correct` 一律判 0。这系统性惩罚**会改写措辞的方法**:多 agent workflow 的最后一个决策节点常常用自己的话重述答案,而单链方法直接吐 `\boxed{}`。测到的不是能力差异,是格式巧合。

两端同时修:

1. **格式要求 + 示例**(`shared/bench.py: ANSWER_FORMAT`),追加在 `question_text` 后面。`question_text` 是七个方法唯一收到完全一致的那段文本,所以在这里加是均匀的;改任何方法自己的 prompt 都会改动被测对象本身。
   示例**只演示格式**:不含推理、不含解题步骤、不含领域提示,所以没有方法从中获得解题优势。
2. **分层宽松抽取**(同文件),按显式标记 → 结构线索 → 短尾行逐级降级,每一层单独计数,所以「格式合规率」可以作为一列单独报告,而不是混进准确率里。

现状:24/24 判对。

### 1.4 LaTeX 排版差异的二次判定(仅 MATH/AMC)

AFlow 的 `math_equal` 依次尝试:精确字符串 → 数值 `isclose` → sympy 符号相等。**坐标对和区间三层全过不了** —— `parse_latex` 解析不了 `(3, \frac{\pi}{2})` 这种非表达式,于是模型答对了也判 0。实测我们的 MATH 划分里 26/500(5.2%)的标准答案带这类纯排版差异(`\left/\right`、`\dfrac`、间距宏、`\text{}`),而方法间差距只有 1–4 分。

处理方式是**在作者判分给 0 之后再重试一次**,双方同时做排版归一化(去 `\left/\right`、`\dfrac`→`\frac`、去间距宏、去 `$`、去全部空白),命中则计 `recovered_latex_form`。

刻意做成「重试」而不是「替换」:作者的判定优先且**永不被下调**,所以这一层只能救回分、不可能拿掉分。测试:7 种排版差异全部救回,8 种真错答案(分母不同、坐标颠倒、符号相反、单位不同、开闭区间不同)**0 误判**。

### 1.5 全量 input/output 落盘

七个方法的每一次 LLM 调用都经过代理,所以代理是唯一能覆盖「所有方法所有环节」的单点。`logs/transcripts.jsonl` 记完整 messages + completion,以 `request_id` 与 `logs/api_calls.jsonl`(延迟、token、截断标记、namespace)一对一 join。实测 1420–3100 字节/次,全量 sweep 约 2 GB,上限 20 GB 后停止记录并打印告警(**不影响判分**)。

### 1.6 协议指纹与新旧结果隔离

改了格式要求就改了模型看到的输入,改了抽取层就改了判分 —— 两者产生的分数不能放进同一张表。两道防线:

* `separate_runs.py` 把上一轮协议的全部产物**移动**(不是删除)到 `archive/<label>/`,新跑从空目录开始。有活进程时拒绝执行。旧跑仍是若干结论的唯一证据(例如实测的预算不对称:FlowBank 在 DROP 上 10.5M token vs CARD 313K),而且留着不再花 GPU 时间。
* `bench.protocol_fingerprint()` 给出 prompt 与 scorer 两个哈希,`sweep.py` 逐作业写进 `protocol.json`,`collect.py` 比对后**拒绝报告**指纹不符的作业。第二道防线存在的理由是第一道依赖我把七个仓库、九个方法条目的落盘位置数全 —— 这种清单恰恰最容易漏。


### 1.7 采样协议:presence_penalty 1.0 + 撞满上限则整请求重跑一次

**问题。** 温度 0 下 Qwen3-8B 会陷入复读。实测(旧跑的全量落盘):撞满 8192 token 上限的回复里 **92.9% 是退化的**(同一行/短语反复),只有 5.2% 是真的长答案。也就是说"撞满上限"基本等价于"解码进了循环",而不是"题目需要长回答"。

**两个被否掉的做法。**

* **续写**(旧协议):实测 88.8% 能救回,但拼出来的回复是"截断的一段 + 另一次不同条件下的生成",被判分的东西不是方法真正产生的回复。用户明确否掉:"我不能接受续写"。
* **当成精度问题去调抽取层**:方向错了 —— 病因在解码,不在抽取。

**采用的做法**,依据 Qwen3 模型卡("不要用 greedy decoding";presence_penalty 0–2 缓解无尽重复,过高会引起语言混杂)加实测:

| 设置 | 仍复读的比例 | 受影响题目的准确率 |
|---|---|---|
| 现状(penalty 0,续写) | 基线 | 0.355 |
| presence_penalty = 1.0 | 清掉 93% | — |
| + 撞满则重跑 1 次 | 剩 2% | **0.654** |

判定条件就用 `finish_reason == "length"` 本身,不额外做复读检测 —— 更便宜也更不容易被质疑。重跑是**同一请求的全新生成**,所以被判分的永远是一段完整、未被拼接的回复;它之所以有效,是因为 vLLM 的 continuous batching 在温度 0 下也不是逐位可复现的,重发会落进不同的 batch、走不同的路径。**只重跑一次**:第二次重跑的边际回收实测不足一个百分点,而每次重跑都是成倍的成本。

`max_tokens` 保持 8192(不下调 —— 调小只是让模型更早撞墙,并不会让它答得更短)。续写代码保留但默认关闭,以便按需复现旧协议。

**顺手修掉的计数错误:** 旧代码在续写请求返回**之前**就把 `recovered` 置真,于是报出"100% 恢复",真实值是 88.8%。现在只有在重跑确实返回了未截断的回复时才置真,并新增 `hit_length_once`(撞满过几次)与 `retry_also_truncated`(重跑仍复读)两个独立计数,以及 `retry_completion_tokens`(重跑的 token 单独计,不并入原回复,保证成本可归属)。

`test_proxy_protocol.py` 直接对 `_normalize` 建出的 payload 和替换掉 `_forward` 的重跑分支断言,20 项全过 —— 包括"调用方显式传 presence_penalty=0 也不能关掉它"。

### 1.8 模型生成代码的 30 秒硬超时(四个执行代码的仓库一视同仁)

**问题。** 三个仓库都无法真正停住自己生成的代码:

* **MaAS / DAAO** `Programmer.exec_code`:`with ProcessPoolExecutor(...)`,超时后 `shutdown(wait=False, cancel_futures=True)` —— 这**杀不掉已经在跑的子进程**,紧接着 `with` 退出时调用 `shutdown(wait=True)`,于是**永久等待**。标称超时 600 秒,实际行为是死循环把作业挂死。
* **AFlow** 同一段代码(标称 30 秒,同样永久挂死),外加 `Test.exec_code` 里 `exec(test_code, globals())` **完全没有超时**。
* **FlowBank** 是唯一撑得住的:长生命周期进程池在 `with` 之外,逐用例 30 秒超时后重建池 —— 代价是泄漏一个空转进程,但作业不死。

**修法。** `shared/exec_guard.py`:代码送进一个**独立解释器子进程**(`sys.executable -c`,stdin 喂参数),超时后由父进程 `subprocess.run` 硬杀。为什么不用另两种机制:线程杀不掉(`while True: pass` 会占着一个核直到跑完,池的 shutdown 还会阻塞);`signal.alarm` 只在字节码之间触发,拦不住 C 层循环(`sum(1 for _ in itertools.count())` 完全无视它)。也没有用 fork —— 父进程是带线程的 asyncio 程序,fork 会连同别的线程持有的锁一起复制,继承了 logging 锁的子进程会死锁,而死锁正是这个模块要消灭的东西。

30 秒的依据:实测全部记录里**最慢的正确解是 3.16 秒**,一个数量级的余量且有界。

**保留的作者行为:** `run_code` 里的禁止导入清单(os / sys / subprocess / matplotlib 等 14 项)与拒绝消息在安装时从各仓库自己的源码里抽出、注入到防护里,逐字保留 —— 绕过它就等于改变了算子拒绝执行什么。算子的控制流也没动:防护把失败断言重新抛成 `AssertionError`、超时抛成 `TimeoutError`,所以每个作者自己的 except 分支照原样分类结果。

**过程中被测试抓到的一个 bug:** 子进程原本把结果 JSON 写 stdout,而 MBPP 的解答经常 `print` —— 一个打印 100 kB 的解答会把 JSON 冲烂,导致整批结果都判成失败。改为结果写临时文件、生成代码的 stdout/stderr 重定向到 devnull。`test_exec_guard.py` 13 项全过(含 Python 层死循环、C 层死循环、杀后无残留进程、挂死之前的用例仍保留判定)。

### 1.9 命名空间细化到(阶段,方法,数据集),指纹纳入采样参数

代理的计费命名空间取自 URL 路径,而七个仓库各自把 URL 写在自己 checkout 里的一个文件中 —— 于是 DAAO 的四个数据集、两个阶段全都记在 `train/daao` 一个桶里,"哪个格子花了多少算力"和"哪个格子不再发请求了"都无法回答。

两条通路,免改仓库或只改一处:

* **G-Designer / CARD / MasRouter** 读 `os.getenv("BASE_URL")`,而 python-dotenv 默认不覆盖已存在的环境变量 —— 所以 `sweep.py` 导出 `BASE_URL` 就压过了 `.env`,这三个仓库一个字没改。
* **MaAS / DAAO / AFlow / FlowBank** 走 YAML 加载器、没有环境变量覆盖,但各自只有一个 `create_llm_instance(config)`,是所有 provider 的唯一构造点(包括优化器自己那个与执行器不同的 config 对象),在那里认 `SHIM_BASE_URL` 即可全覆盖。未设该变量时行为与从前完全一致。

现在 URL 形如 `http://127.0.0.1:18080/search/daao/mmlu_pro/v1`。`test_namespace.py` 在七个仓库各自的 venv 里逐个验证通过。

**指纹补齐:** `protocol.json` 现在还记录 `sampling`(temperature / presence_penalty / max_tokens / max_model_len / enable_thinking / 撞满后的处理规则)。这不是细节 —— 换了 penalty、换了上限、换了撞满后的处理,模型产出就不同;缺了这一项,"续写 + 无 penalty" 的跑和"penalty 1.0 + 重跑"的跑在 collector 眼里一模一样。

### 1.10 卡死检测与栈转储

`watchdog.py` 判定一个作业卡死需要**两个条件同时成立**:日志超过 45 分钟没长,且该命名空间在同一窗口内没有完成过任何请求。单看任一个都会误报 —— 优化器的一步长计算日志是安静的但仍在发请求;在两次调用之间做本地计算的作业在网络上是安静的但仍在写日志。

判定为卡死后向该作业发 `SIGUSR1`,栈落到作业自己的目录。`PYTHONFAULTHANDLER=1` 只管致命信号,管不了活着的挂死,所以 SIGUSR1 的处理器由 `shared/pyhooks/sitecustomize.py` 注册 —— 把该目录放进 `PYTHONPATH`,Python 启动时自动导入,七个仓库一行代码都不用改。本机 `kernel.yama.ptrace_scope = 1` 禁止 py-spy / gdb attach 到非子进程,这是唯一可行的路子。

进程匹配读 `/proc` 而不是用 `pgrep -f`：`pgrep -f` 会匹配到调用它的那条命令行本身,第一版就因此对已经退出的作业报出"1 pid"。报告也据实而言 —— 只有转储文件真的变大了才说"栈已抓到",否则明说"信号发了但文件没长"。

### 1.11 并发上限是实测的,不是猜的

`concurrency_probe.py` 回放 `transcripts.jsonl` 里**真实记录的 prompt**(不是合成的),按客户端并发 16/32/64/128/192 各压 90 秒:

| 并发 | 解码吞吐 | 平均延迟 | vLLM Waiting |
|---|---|---|---|
| 16 | 519 tok/s | 4.3 s | 0 |
| 32 | 811 tok/s (+56%) | 4.9 s | 0 |
| 64 | 1239 tok/s (+53%) | 7.2 s | 0 |
| 128 | 1558 tok/s (+26%) | 9.5 s | 0 |
| 192 | 1616 tok/s (**+3.7%**) | 13.3 s | 0 |

第一版探针是错的,记下来免得重犯:题目是"17×23",模型四个 token 就答完,四分钟跑了 105,311 次请求 —— 测的几乎全是 prefill,与真实负载(几百 token 的推理回复)毫不相干,按它定并发就是按错误的瓶颈定。

`job_concurrency.py` 再从 `api_calls.jsonl` 的起止时间算出旧跑(`--jobs 12`)23 小时内**平均在途 69.7**、峰值 247,即只用到实测上限的 54%,约有 1.8× 余量 —— 于是新跑取 `--jobs 16`(对内存保守),实测在途稳定在 128,正落在拐点上。

未提高 `--max-num-seqs`:两块 A800 与他人共享,我们每实例只拿到约 780 tok/s(独占时 Qwen3-8B 可达 2000+),而 64→128 已只涨 26%(次线性),说明瓶颈是被共享的 GPU 算力而非我们的 seq 上限;抢占计数也已非零(50 / 31)。

**顺带修掉的一个真实浪费:** 16 个作业的 torch/OpenMP 按机器核数开线程池,128 核的机器上 load average 冲到 218。这些客户端模型极小(拓扑 GCN、难度 VAE、MiniLM),且工作量本来就被两个 vLLM 实例卡住,线程什么也没买到,还拖慢同机另外八个人。限成 `OMP_NUM_THREADS=4` 后 load 降到约 105。

---

## 2. 与原论文不同、必须在论文中申报的项

| # | 项 | 论文设置 | 本实验 | 原因 |
|---|---|---|---|---|
| 1 | Executor | GPT-4o-mini(FlowBank 表内所有方法统一) | **Qwen3-8B 关 thinking** | 成本;上一轮 API 欠费中断 |
| 2 | Optimizer | AFlow 双臂 GPT-4o / Qwen3-8B;ADAS、AgentSquare 用 GPT-4o | 统一 Qwen3-8B | FlowBank 已把「AFlow + Qwen3-8B optimizer」作为正式 baseline,有先例 |
| 3 | 数据划分 | 各论文自定 | **上游规范划分**,3 个全量 + 2 个固定种子(seed 20260821),带 sha256 | 见 §2.1 |
| 4 | AMC 搜索集 | — | **无。改用 MATH 搜索集迁移** | 规范 AMC 仅 83 题,全部用于评测,无 train 划分可切 |
| 5 | FlowBank portfolio | 作者 Stage-1 产物 | **不复用,三阶段在规范划分上重跑**(修正本文档早期结论) | 审计发现其划分与我们的测试集大量重合:按 `task_id` 比对,MBPP 有 **203/500** 项相同(按文本比对显示 0,是 prompt 格式不同造成的**系统性低估**)。在与测试集重合的划分上筛出的 portfolio + 训好的 selector,等于把测试数据泄漏给一个待排名的方法 |
| 6 | G-Designer/CARD 的 DROP domain | 无(仓库无阅读理解 domain) | **继承 `mmlu` prompt set** | 其角色为通用问答;若继承 `gsm8k`(角色为「数学分析师」)会因无关原因削弱该方法 |
| 7 | ScoreFlow | FlowBank 的 baseline 之一 | **不复现** | README 要求 80–90 GB 显存,本机 2×32 GB 凑不出 |
| 8 | torch / vLLM | 各论文自定(MaAS pin `torch==2.1.0+cu118`,FlowBank pin `torch 2.1.0`+PyG 2.7.0) | torch 2.10.0+cu128 / vLLM 0.19.1 | 驱动 570.124.06 上限 CUDA 12.8;详见家目录 A800 环境约束 |

### 2.1 数据划分明细

| 数据集 | 来源 | n | 抽样 |
|---|---|---:|---|
| MATH | `HuggingFaceH4/MATH-500` [test] | 500 | 无(全 5 难度、全 7 学科) |
| AMC | `AI-MO/aimo-validation-amc` [train] | 83 | 无 |
| MBPP | `google-research-datasets/mbpp` [full/test] | 500 | 无(task_id 恰为官方 11–510) |
| DROP | `ucinlp/drop` [validation] | 1000 | 均匀随机,seed 20260821 |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` [test] | 1120 | 14 学科 × 80 分层,seed 20260821 |
| | | **3203** | |

搜索集:`math_search` 256(hendrycks_math train)、`mbpp_search` 256、`drop_search` 256、`mmlu_pro_search` 252(取自 MMLU-Pro test 中**与评测集不相交**的部分,实测重叠 = 0)。

**未采用 FlowBank 自带 jsonl 的原因**(审计结果):其 `math_test.jsonl` 全为 Level 5 且只覆盖 7 个学科中的 4 个;`mbpp_test.jsonl` 仅 60% 落在官方 test 区间,其余来自 train/val 乃至 few-shot 示例区(含 task_id 2、3)。这两点仓库中均无文档说明,且与任何已发表 MATH/MBPP 数字都不可比。其 test/validate 无泄漏、MMLU-Pro 为严格 80×13 分层,这两项是干净的。

### 2.2 提示词按数据集适配(七个方法逐个核对,改完与原始逐行 diff)

我们给这些仓库加了它们原本不支持的数据集,而每个仓库的提示词都写死了它自己那几个数据集的任务身份。于是 DROP 的题目被要求"solve the given mathematical problem"、MMLU-Pro 被告知"只有 A、B、C、D 四个选项"、MBPP 被要求把答案写进 `\boxed{}`。这类措辞不是方法的设计,是数据集的身份声明写错了。

**改动原则(全程遵守,可用 `prompt_baseline.py --diff` 复核):** 只改任务身份 —— 这是什么任务、答案该长什么样、示例是什么;角色、推理步骤、输出格式机制、辩论结构、`SC_ENSEMBLE_PROMPT`、`SELFREFINE_PROMPT`、词数限制、首行规则一律不动。改动前先 `prompt_baseline.py --save` 存下 94 个提示词文件与 sha256 清单。

#### 缺陷修复(与适配分开记账):MaAS / DAAO 的 LaTeX 转义损坏

作者的模板是**非 raw 的三引号字符串**且内含 LaTeX,于是 `\boxed` 在运行时变成单字节 `0x08` 加 `"oxed"`,`\frac` 变 `\x0crac`,`\times` 变 `\x09imes`。文件在编辑器里、在 `git show` 里看都是对的 —— 只有运行时的值是坏的。`scan_escape_corruption.py` 实测:**MaAS 92 处、DAAO 92 处,其余五个方法零处**。也就是说只有这两个方法在告诉模型"present the final answer enclosed in ␈oxed{} LaTeX notation",而判分器找的是 `\boxed`。

这不是设计改动,是把作者本来写的字面文本恢复回去;但它也确实不属于"数据集适配",所以单列一类申报。修复对所有数据集生效(MATH/AMC 同样受损)。

**扫描器本身也修了一次值得记的错:** 第一版用 `ast.literal_eval` 静态读取,而修复是在导入时由追加块完成的 —— 于是它对运行时已经干净的文件报"92 处损坏"。改为真正 `exec` 模块后归零。这正是这个扫描器存在的目的所在的那类错误。

#### MaAS / DAAO:DROP、MMLU-Pro、MBPP 的措辞

**先查清了一个陷阱:** 五个 `SHARED_*` 工作区的 graph 全都 `import maas.ext.maas.scripts.optimized.MATH.train.template.prompt`(MBPP 那个 import `HumanEval`)—— 它们自己目录下的 `prompt.py` 副本**从来不被导入**。所以改那些副本等于没改;必须像 MBPP 的 `dataset="HumanEval"` 那次一样,在共享模板里按运行时数据集选择。

实现方式是往共享模板**追加**一个按 `SHIM_DATASET` 分派的覆盖块,而不是原地编辑 —— 于是作者的原文留在上方可读,且 diff 可证。适配内容:

* DROP:`GENERATE_SOLUTION_PROMPT` / `MATH_SOLUTION_PROMPT` / `REFINE_ANSWER_PROMPT` / `SOLUTION_PROMPT` / `DETAILED_SOLUTION_PROMPT` / `MATH_SOLVE_PROMPT` 的"mathematical problem"改为段落阅读理解,末行格式改为 `Answer: <answer>`(判分器读的就是这个),编号条目的**数量与顺序保持原样**;`PYTHON_CODE_VERIFIER_PROMPT` 只改"基于什么写代码"这一句,`solve` 契约与占位符不动;`GENERATE_COT_PROMPT` 的 GSM8K/MATH 两个示例换成同样形状的两个 DROP 示例。
* MMLU-Pro:同上,改为十选一措辞、末行 `Answer: (X)`。选项数**不写死为十** —— 本划分里各题选项数不同,写任何一个固定数都会对一部分题为假。
* MBPP:走 HumanEval 模板,`IMPROVE_CODE_PROMPT` 与 `GENERATE_CODE_PROMPT` 里的"HumanEval benchmark/dataset"改为 MBPP;该模板其余部分是通用的。
* MATH / AMC:除上述缺陷修复外**逐字未动**。GSM8K 风格的少样本示例保留 —— 任务类型本就匹配。

`--check` 断言:五个数据集下所有提示词常量零控制字符;MATH 仍含作者原措辞且 `\boxed{}` 可读;DROP/MMLU-Pro 不再含"mathematical problem"也不含 `\boxed`;`SC_ENSEMBLE_PROMPT` 与 `SELFREFINE_PROMPT` 在三个数据集下逐字节相同;Programmer 契约与占位符完好。

`prompt_baseline.py --diff` 结果:**64 个文件变化,全部为纯追加(0 处删除/修改行)**,即作者原文逐字节未动。

#### G-Designer / CARD:选项数与 DROP 的答案后处理

`mmlu` 域在四处写死了四选项(`get_constraint`、`get_analyze_constraint`、`get_decision_constraint`、`get_adversarial_answer_prompt`),而 DROP 与 MMLU-Pro 都继承它。

**先量化再动手,结论与我的假设相反,如实记录:** `letter_space.py` 统计出 5,488 条 MMLU-Pro 和 5,024 条 DROP 的调用带着"只有 A、B、C、D 四个答案"这句话,但它**没有约束住模型** —— 最终答案落在 E–J 的占 52.5%,而 gold 落在那里的是 53.9%;DROP 的最终决策也照样返回真实片段("Answer: Corey Dillon")而不是字母。模型在矛盾中选择了服从任务正文(`shared/bench.py` 追加的那段)。所以这一项**预期的分数影响很小**。

仍然改,理由两条:提示词里断言了关于数据集的假事实,这样的协议不该写进论文;而"模型恰好忽略了它"是这个执行器的性质,不是保证。

做法是**子类覆盖**(`shims/gdesigner_family/shared_prompt_sets.py`),作者文件零改动。只有那四句选项计数改了;角色表、角色描述、连接图、词数限制、首行规则、"可参考其他 agent 答案"一句、决策角色全部继承 —— 连 `get_analyze_constraint` 里作者那个运算符优先级(角色已知时只返回角色描述、把约束文本丢掉)也照原样复制,没有"顺手修好"。DROP 另外覆盖 `postprocess_answer`:继承版最后一行 `answer = answer[0]` 取首字符,对字母是对的,对片段会把 "Corey Dillon" 变成 "C"。

`render_prompt_sets.py` 在两个仓库各 19 项断言全过,包含"math/amc/mbpp 三个域的措辞逐字继承"与"已知角色路径与作者逐字节一致"。

#### MasRouter:最终决策节点的提示词 + DROP 角色池

**发现的真正污染比选项数严重得多。** 作者按数据集切换最终节点提示词(`run_math.py` 用 `math.json`、`run_mmlu.py` 用 `mmlu.json`、`run_mbpp.py` 用 `mbpp.json`),而我们的 `run_shared.py` 是从 `run_math.py` 派生的,继承了 `math.json` 并用于全部五个数据集。FinalRefer 产生的正是被判分的那个答案 —— 于是**每一个 MBPP 答案都被要求写成"the answer is \boxed{...} without any units"而不是 Python 代码块**,MMLU-Pro 和 DROP 的答案则被要求写成带框的数学表达式。

修法用作者自己的资产:按 `--shared_dataset` 选文件,math/amc 用作者的 `math.json`、mbpp 用作者的 `mbpp.json`;mmlu_pro 与 drop 用从作者 `mmlu.json` 派生的两份 —— system 行、决策框架、`\boxed{}` 输出格式、"末行只有一句"规则全部逐字保留,只改选项计数 / 答案形状(判分器的 mmlu_pro 抽取有 `boxed` 层,所以作者的输出格式不需要动)。

**任务类型没有加第四个。** MasRouter 把每个 query 分类到它自带的三个类型之一,再按类型名从 `MAR/Roles/<Type>/` 载入角色池。加第四个类型会让任务分类器多打一个分,连 MATH 和 MBPP 的路由分布一起改变,污染本来正确的格子。DROP 走 Commonsense —— 这正是作者自己 `run_mmlu.py` 对 MMLU 用的映射(`task_labels = [1]`),所以这个替代是他们已经做过的,不是我们新造的。

DROP 需要的只是措辞:七个 Commonsense 角色里有三处答案形状对片段任务是错的("Please analyze step by step and choose the correct answer." 两处、"You will be given a complex math problem ." 一处)。派生出 `MAR/Roles/Commonsense_drop/`,**只改这三句**,其余四个角色、`MessageAggregation`/`OutputFormat`/`PostProcess`/`PostDescription` 字段、角色数量与名字全部照抄;`encoder_roles` 加一段:存在 `<Type>_<dataset>/` 时用它替换该数据集的池,类型列表本身不动,所以其他数据集的路由分毫不移。

#### AFlow / FlowBank:核对后确认不需要改

不是"看着通用就放过",而是按前两次踩过的模式(共享模板、默认值指向别的数据集)逐项查:

* **AFlow**:五个 `SHARED_*` 工作区各自 import 自己的 `template/operator` 与 `round_1/prompt`,不存在交叉引用;逐工作区扫任务身份词,`SHARED_MATH`/`SHARED_AMC` 的 op_prompt 含数学措辞(对它们正确),`SHARED_MBPP`/`SHARED_DROP`/`SHARED_MMLUPRO` 的**一个任务身份词都没有**(通用)。
* **FlowBank**:五个 `SHARED_*` 都是从作者**原生对应**工作区种子的(DROP→DROP、MMLU_Pro→MMLU_PRO、MBPP→MBPP),`operator.py` 与作者版逐字节一致(只有 `operator.json` 因我们的算子注册而不同)。它原生支持这些数据集。

---

## 3. 逐仓库修改

### 3.0 ScEnsemble 字母空间修复(AFlow / MaAS / DAAO,仅 MMLU-Pro)

作者的 `ScEnsemble` 把候选解标成 A/B/C 让模型选一个字母。对 MATH、GSM8K、HumanEval 这毫无歧义 —— 那些数据集的答案**不可能是一个孤零零的字母**。但 MMLU-Pro 是十选一,选项本身就叫 (A)–(J),两个字母空间重叠了:模型回 `E` 指的是题目的选项 E,算子当成第 5 个候选去索引(只有 3 个),`solutions[answer_mapping["E"]]` 抛 KeyError。

后果不是「选得不好」而是**样本丢失**:异常穿透到我们的 shim,被记成零梯度占位。实测(ICL 前的归档,ICL 后同一量级,**与本次协议改动无关**):

| | 丢样本 | 已跑 | 占比 |
|---|---|---|---|
| MaAS / MMLU-Pro | 29 | ≈56 | **>50%** |
| DAAO / MMLU-Pro | 43 | ≈80 | **>50%** |
| MaAS / DAAO 在 DROP | 0 | 同等日志量 | 0 |

**FlowBank 的作者已经修过这个问题**,而且就是为 MMLU-Pro 修的:`DiverseFlow/workspace/MMLU_Pro/workflows/template/operator.py`(其 commit `dde948d`,作者 `lingzhiy`)把候选标号换成 1/2/3,并加了越界校验与兜底,docstring 明写 *"to avoid conflict with multiple-choice answer options A-J"*。也就是说 FlowBank 参赛时自带这层防护,另外三家没有。

所以这里做的是**把对比中已有的一份作者写法一致化**,不是发明新改动 —— 而且 MMLU-Pro 本来就是我们给这些仓库补进去的数据集(AFlow 的 MMLU-Pro 支持整个是本项目的 shim),字母冲突是这次接入带来的副作用。

实现(`shims/maas_family/install.py`、`shims/aflow/install.py`):以**追加子类**的方式覆盖,作者的类原样保留、不改一行,对 upstream 的 diff 是新增而非重写。

两条刻意的边界:

* **数字标号只在 MMLU-Pro 生效**(那是唯一发生重叠的数据集)。MaAS/DAAO 的算子模板跨数据集共用,所以判定放在运行期:先读 `SHIM_DATASET`(`sweep.py` 逐作业注入),再退回 argv,这样脱离 sweep 手工跑 `--dataset SHARED_MMLUPRO` 也不会静默退回字母。AFlow 的模板按 workspace 分开,只需改 MMLU-Pro 那一份,无需运行期判定。
* **越界兜底在所有数据集生效**。这不可能改变本来正常的跑法 —— 它只拦截当前会抛异常的输入。兜底走 FlowBank 原版的阶梯(从啰嗦回复里抽标号 → 退回第一个候选),并且**打日志**,因为退到最后一步意味着「选择」这一步事实上失效了,属于真实降级,不该静默。

验证:MaAS/DAAO/AFlow 三家各 4–6 个场景(数字回复、越界字母 `E`、啰嗦 `solution 3`、空回复、字母回复 `A`、越界字母 `Z`)全部 0 崩溃,且实测下发的标号空间正确(MMLU-Pro=数字,MATH=字母)。开跑后 mmlu_pro 四个作业的 `sample_failed`、`KeyError`、兜底次数**全为 0**。

### 3.1 AFlow `3f45721`

| 文件 | +/− | 性质 |
|---|---|---|
| `run.py` | +35 / −1 | 追加 5 条 `EXPERIMENT_CONFIGS`;`download(["datasets"])` 包进 try/except |
| `scripts/evaluator.py` | +4 / −1 | `dataset_configs.update(SHARED_DATASET_CONFIGS)`;`DatasetType` 扩充 |
| `benchmarks/shared_benchmarks.py` | 新增 | 5 个共享 benchmark 类 |
| `data/datasets/shared_*.jsonl` | 新增 10 个文件 | 按字段映射写出(非符号链接) |
| `_scorer_probe.py` | 新增 | 仅供依赖解析器一次性发现评测器依赖,不参与实验 |

**判分被覆盖 —— 这一条修正了本文档早期版本的说法。** 起初判断 AFlow「只需数据 + 注册,判分无需 shim」,因为共享层调用的正是它自己的评测器。核对后发现:MATH / AMC / MBPP 三个,作者的 `evaluate_problem` 与共享层走的是同一条 `calculate_score` 路径,结果一致;**但 DROP 分叉** —— AFlow 把回复按 `|` 切片、对每片算 F1 取最大,而共享层先抽取所述答案再算。两者数字不同,若不统一,DROP 一列会出现「AFlow 用一套判法、其余六个方法用另一套」。

因此 5 个数据集的 `evaluate_problem` 全部改走共享层。**指标定义仍是作者的**(共享层底层调用其 `calculate_score` / `check_solution`),统一的只是答案抽取与聚合。已验证:一个含 30 遍填充文本 + `Answer: X` 的啰嗦回复经抽取仍得 1.0,而 AFlow 原生路径会判成接近 0。

**另外两点**:
- 仓库自带 `benchmarks/amc.py` 但 **AMC 从未被注册进 `EXPERIMENT_CONFIGS`**,即在本实验之前就是不可达的。
- `download(["datasets"])` 指向 Google Drive,本机不可达且会中断每次运行,故改为捕获异常后继续(数据由 shim 提供)。
- AFlow 的 `amc.py` 读 `problem["question"]`,而规范划分里该字段名为 `problem`。**数据以写出而非链接的方式生成**,在 shim 内补 `question` 别名,规范数据与其 sha256 保持不动。

**本轮新增的三项(均为跑通闭环所必需)**:

- **工作区播种**:AFlow 靠改写既有工作流来优化,每个数据集需要 `workspace/<KEY>/workflows/{round_1,template}`。缺失时**不会失败**——日志打一句 `No module named workspace.SHARED_MATH.workflows.round_1.graph`,每轮判 `None`,然后像正常跑完一样退出。五个共享数据集各从最接近的作者工作区播种(MATH/AMC←MATH、MBPP←MBPP、DROP←DROP),**MMLU-Pro 无对应,借用 HotpotQA**(同为 QA,算子集同为 Custom/AnswerGenerate/ScEnsemble)——属申报的替代。`graph.py` 用绝对模块路径 import 算子,故播种时重写 import 前缀,否则每个数据集会静默执行 MATH 的算子和 prompt。
- **补齐算子的构造签名(替代了早先的"放宽重试预算")**:传给优化器的算子描述只写调用签名(`sc_ensemble(solutions, problem) -> dict`),从不写构造形式。强优化器看不出问题,弱优化器致命。

  证据是一次受控对照:**算子描述完全相同**的情况下,同一个 Qwen3-8B 在 FlowBank 的 DiverseFlow 里写出正确的 `operator.ScEnsemble(self.llm)`(其种子工作流里原本就有这一行),在 AFlow 里写成 `operator.ScEnsemble()`(其 MATH 种子只实例化了 `Custom`)。模型被要求从**单个**范例泛化构造约定,而它做不到。于是每个新增算子的轮次都以 `TypeError` 判 `None`,AFlow 上报的就是种子工作流的分数。

  实测(作者默认 `max_retries=1`):补齐前 round 2/3/4 全部 `None`、3 次失败记录;补齐后 round 2 生成 `operator.ScEnsemble(self.llm)`、**0 次失败**。

  **选择补文档而非改种子工作流**:工作流是方法的起点、其分数必须保持可比,而这是一个自称"给出 interface"的字段里的事实性缺失,且不暗示应该搭出什么样的工作流。签名**从算子源码推导**,推导不出就报 FAIL —— 写错的构造签名比不写更糟。AFlow 与 DiverseFlow **同等施加**(16 / 25 条),避免变成对某一方的偏袒。

- ~~**重试预算 `AFLOW_MAX_RETRIES` 默认 3**~~ —— **已撤回,默认恢复作者的 1。**

  当初把默认值设为 3 是**拍的,没有推导依据**,而且我引用的证据是错的:那次跑里 round 2/3 都是第一次尝试就成功,记录的 2 次失败是代理上下文裁剪的 bug(确定性失败,重试帮不上),我把因果搞混了。根因定位后重试也不再需要 —— 它本来也是个坏办法,因为那不是随机失败而是同一个推断错误。env 旋钮保留为可申报的杠杆,**失败记录始终开启**,无效图率是我们上报的量。

- **安装器自身的编译校验**:早期版本用 `re.subn` 的模板串写 `\"datasets\"`,反斜杠被原样写进文件,`run.py` 带着 SyntaxError —— 而 `--check` 只查标记字符串在不在,**一直报 OK**。现在凡安装器改过的 `.py` 都必须 `ast.parse` 通过。

### 3.1b MasRouter `e005f76`

| 文件 | 性质 |
|---|---|
| `Experiments/run_shared.py` | 新增,由 `run_math.py` 正则派生,**diff 28 行** |
| `Datasets/shared_dataset.py` | 新增,喂数 + 判分 + 任务标签 |
| `Datasets/shared/*.jsonl` | 新增 9 个符号链接 |

**任务标签是这个仓库的关键点**:MasRouter 用 `F.cross_entropy(tasks_probs, tasks_y)` 训练一个任务分类器,taxonomy 为 Math / Commonsense / Code 三类。`run_math.py` 里该标签硬编码为 `0`(Math)。若照搬,路由器会被教成「所有查询都是数学题」,路由决策拿不到有用梯度,而准确率数字看起来完全正常。映射:math/amc → Math(0),mbpp → Code(2),drop/mmlu_pro → Commonsense(1)。

**判分与任务标签各出现两处**(训练循环与测试循环),替换必须都覆盖 —— 漏掉第二处会让上报数字无论跑哪个数据集都由 MATH 的判分器给出。安装器对这两项断言「必须匹配 2 次」。

**本轮新增(此前该仓库从未真正发出过一次请求)**:

| 改动 | 原因 |
|---|---|
| `MAR/LLM/llm_profile.py` 由 5 个云模型收敛为 1 条如实描述本地模型的条目 | 见 §1.1 第 5 条。**代价**:LLM 路由头变成单选,log-prob 恒为 0、不产生梯度,MasRouter 四个轴丢掉一个。但其 loss 为 `task_loss + answer_loss + 0.001·vae_loss`,**无 cost 项**,该头本来只能用准确率换价格,故在本文排序所用的准确率指标上不构成劣势。实测 task/collab/数量/role 四个头仍在学(reasoning 模式跑出 Chain/CoT/Debate/FullConnected/IO/Reflection 六种,agent 数在 3–4 波动) |
| `.env` 新增 `URL` / `KEY`(以及 `BASE_URL` / `API_KEY`) | `llm_registry` 把非 DeepSeek 的模型名都交给 `ALLChat`,后者用官方 OpenAI SDK 读 `os.environ["URL"]`。缺失时 base_url 为 None → 打公网 API,外面套着 `@retry(指数退避, 10 次)`,表现为**十余分钟静默卡死、日志无一行、代理侧零请求**。定位靠单独调一次 `agen`(立刻报 `Missing credentials`),而非读代码推断 |
| `MAR/LLM/llm_embedding.py` 返回 `embeddings.clone()` | sentence-transformers ≥3 的 `encode` 在 `torch.inference_mode()` 下运行,产出张量不能 saved for backward;该 repo 按 2.x 写的。这些 embedding 直喂路由器的可训练头,首个 batch 即崩 |
| `MAR/LLM/price.py` 注册 `qwen3-8b`(0 价) | `cost_count()` 对未知模型名 `return 0,0,0`,而且**返回在 token 计数之前**,不注册会连 prompt/completion token 一起清零 |
| `Experiments/run_shared.py` 新增 `--max_batches` | 训练循环无早停手段,最小冒烟单位是整个 epoch。默认 None,不影响正式 sweep。**注意其测试循环不受此参数限制** |
| `MAR/LLM/gpt_chat.py` 去掉 `model + '-y'` 后缀(2 处) | 作者内部网关的 tag。**这条在 `GPTChat` 路径上,而 registry 不会为我们的模型名选中它,所以对当前运行没有影响** —— 之所以照改,是因为 `cost_count` 已按无后缀名计价,两者不一致正是让 token 记账静默归零的那类陷阱 |

**安装顺序上的一处教训**:`patch_max_batches()` 起初排在 `derive_runner()` 之前,而后者会从 `run_math.py` 重新生成 `run_shared.py`,把补丁覆盖掉 —— 补丁报告 ok(写入确实成功过),运行时 argparse 却不认这个参数。

### 3.2 MaAS `987f3c1`

| 文件 | +/− | 性质 |
|---|---|---|
| `config/config2.yaml` | +17 / −12 | 指向本地代理(`base_url`、`api_key=local`、`temperature=0`);由安装器生成 |
| `maas/ext/maas/benchmark/experiment_configs.py` | +27 / −0 | 追加 5 条 `EXPERIMENT_CONFIGS`;**原有 MATH/GSM8K/HumanEval 三条保留** |
| `maas/ext/maas/scripts/evaluator.py` | +4 / −1 | `dataset_configs.update(SHARED_DATASET_CONFIGS)`;`DatasetType` Literal 扩充 |
| `maas/provider/__init__.py` | +14 / −28 | 13 个厂商 SDK 导入改为容错(见下) |
| `maas/ext/maas/benchmark/shared_shim.py` | 新增 | 泛型 benchmark,判分委托共享层 |
| `maas/ext/maas/data/shared_*.jsonl` | 新增 10 个符号链接 | 指向共享划分 |

**provider 容错的理由**:该文件原本急切导入全部 13 个后端,而 `sparkai` 在 Python 3.12 上无法导入,导致整个仓库不可用。改为逐个 try/except 并据实际可用性重建 `__all__`。本实验只用 OpenAI 兼容后端,**未触碰任何方法逻辑**。

**方法逻辑改动**:无。controller 的 REINFORCE 更新、supernet 采样、Gradient Agent 全部原样。`shared_shim` 忠实返回作者约定的 6 元组 `(input_text, output, expected, score, cost, logprob)`。

**两处上游缺陷(本轮发现,已在 shim 内修复)**:

1. **`GENERATE_COT_PROMPT.format()` 对任何输入都抛 `KeyError`**。作者随附的 ICL 示例里含 `\boxed{-2}`、`\frac{1}{3}`、`\boxed{144\pi}`,`str.format()` 把每个 LaTeX 花括号都当成替换字段。实测 **40/40 道 MATH 全部触发**,即 `GenerateCoT` 与 `MultiGenerateCoT` 两个算子在上游从未工作过 —— 被作者宽泛的 `except` 掩盖成得分 0。改为只替换具名占位符、其余花括号保持字面(`_shim_safe_format`)。
2. ~~**`ScEnsemble` 会因格式不符直接杀掉整个样本**~~ —— **此项已撤回,作者算子恢复原样。**

   曾经的做法:在 XML 字段抽取失败时把 `ScEnsemble` 退化成 `solutions[-1]`。理由是「8B 执行器无法稳定给出候选字母」。**这是同一个症状的第四次误诊。**真正的原因是代理把 `stream=True` 改写成 `False`,MaAS 的 SSE 解析器在纯 JSON 响应体里找不到 `data:` 行、返回空字符串,空字符串在 `ScEnsemble` 内部表现为 pydantic `ValidationError`。

   证据(SSE 直通修好之后):`probe_scensemble.py` 返回 `{'solution_letter': 'A'}`、`fill() error: None`;`ValidationError` 计数从 26(`maas_smoke6`,修复前 22:57)降到 0(`maas_final` 23:42、`daao_smoke` 00:30);用**未打补丁的作者算子**重跑冒烟,ValidationError / 零梯度占位 / 空输出 / KeyError **全为 0**。

   撤回而非"保留但禁用":这是对论文未定义路径上算子行为的修改,在没有证据支持时不该保留。安装器的 `--check` 现在**断言作者算子未被改动**。真实失败仍然可见 —— `shared_shim` 会把样本失败连同点名 ScEnsemble 的 traceback 打出来,故失败率依旧可测,而算子行为未动。

   *清理时的一个坑*:`git checkout` 无法复原退化代码,因为它还存在于 `optimized/SHARED_*/` 这些由 `copytree` 播种出的**未跟踪目录**里(播种发生在补丁之后)。必须删除重新播种。

**cost 项的实证核查**:MaAS/DAAO 的 RL 效用是 `scores - 3×costs`。若 cost 按云端价目表虚构,整个 RL 信号会被扭曲。实测 `Model Qwen/Qwen3-8B not found in TOKEN_COSTS` 在 MaAS 日志出现 194 次、DAAO 31 次 —— `cost_manager` 查不到价格就 `return`(**在累加 token 之后**),故 `total_cost` 恒为 0、token 计数完好,**效用退化为纯准确率**,未被虚构价格污染。MasRouter 的 `utility = is_solved - cost × 100` 同理。

### 3.3 DAAO `5e260bb`

| 文件 | +/− | 性质 |
|---|---|---|
| `config/config2.yaml` | +19 / −0 | 同上 |
| `daao/ext/maas/benchmark/experiment_configs.py` | +52 / −25 | 追加 5 条;**−25 全部来自 CRLF→LF 行尾转换,原有三条完好**(已逐项核验) |
| `daao/ext/maas/scripts/evaluator.py` | +4 / −1 | 同 MaAS |
| `daao/provider/__init__.py` | +14 / −28 | 同 MaAS |
| `daao/tools/schemas/GPTvGenerator.yml` | +5 / −9 | **非人为改动**,见 §4 |
| `daao/ext/maas/benchmark/shared_shim.py` | 新增 | 同 MaAS(自动识别 7 元组) |
| `daao/ext/maas/data/shared_*.jsonl` | 新增 10 个符号链接 | 见下方警示 |

**⚠️ 曾出现并已修复的问题**:安装器最初用 `mbpp_test.jsonl` / `drop_test.jsonl` / `mbpp_train.jsonl` 等无前缀名建链,而 **DAAO 自带这三个同名数据文件**,被静默覆盖。已 `git checkout` 恢复,并把全部键名改为 `SHARED_*` 前缀(数据路径由 `f"{dataset.lower()}_{split}.jsonl"` 派生),现无任何撞名。

**方法逻辑改动**:无。难度 VAE、Agentic Net、Router 全部原样。`shared_shim` 返回 7 元组并**按判分回写 `vae["is_solved"]`** —— 该字段是难度 VAE 的训练信号,漏写会让难度估计器空转而外观正常。

**算子清单不一致(本轮发现)**:DAAO 的 `experiment_configs.py` 列了 7 个算子(含 `EarlyStop`),但它把 `EarlyStop` 从 `operator.json` 删掉、并注释掉了该类 —— 于是图构建用的名字表比优化器构建的 embedding 矩阵短。controller 一旦采到最后一个下标就 `IndexError`(冒烟中 8 个样本里有 3 个),被 DAAO 宽泛的 `except` 报成得分 0。**MaAS 的 json 仍是 7 个**,即两个 fork 确实不同,所以安装器改为**从各仓库自己的 `operator.json` 读取**,而非硬编码(硬编码就等于把这个 bug 抄进来)。

**稀疏检出**:仓库原 1.84 GB,排除 `all-MiniLM-L6-v2/`(931 MB,改为链接共享副本)与 `daao/ext/maas/scripts/optimized/**/*.pth`(906 MB,作者在 gpt-4o-mini 上训的 controller 检查点,本实验需在 Qwen3-8B 上重训,故不需要)。检出后 15 MB。

### 3.4 G-Designer `a6efcfa` — **零受控文件修改**

| 文件 | 性质 |
|---|---|
| `experiments/run_shared.py` | 新增,由本仓库 `run_gsm8k.py` **正则派生,diff 仅 14 行** |
| `datasets/shared_dataset.py` | 新增,喂数 + 判分转接 |
| `GDesigner/prompt/shared_prompt_sets.py` | 新增,注册 5 个 domain |
| `datasets/shared/*.jsonl` | 新增 10 个符号链接 |

**派生而非重写的理由**:`run_gsm8k.py` 内含拓扑的 policy gradient(`utility = is_solved`、`single_loss = -log_prob * utility`、Adam 更新 `graph.gcn`)。重写有改变被测方法的风险,故只替换三处:数据来源、`Graph(domain=...)` 跟随 `--domain`、判分改走共享层。**训练循环逐字节保持作者原样。**

**复用 `--domain` 而非新增参数**:该参数原有帮助文本即为 "Domain (the same as dataset name)",故直接作为数据集选择器,进一步缩小 diff。

**5 个 domain 的继承关系**(全部不覆盖任何方法):`math`/`amc` ← `gsm8k`;`mbpp` ← `humaneval`;`drop`/`mmlu_pro` ← `mmlu`。其中 DROP 属申报项(§2 第 6 条)。

**判分带来的一处语义变化**:原代码 `is_solved = float(pred)==float(true)` 为布尔;共享层对 DROP 返回 F1,故 `is_solved` 可为小数。这使 policy gradient 的 utility 项获得**部分得分**而非被迫二值化。

### 3.5 CARD `d5d1f68`

**注意:与 G-Designer 不同,CARD 有受控文件修改** —— `CARD/llm/llm_registry.py`(新增 `local` provider)与 `CARD/llm/__init__.py`(`together` import 改为可选)。二者都不触碰方法逻辑。

与 G-Designer 完全同构(4 个新增文件,`run_shared.py` diff 同为 14 行)。

**但 CARD 另有六处必须处理(G-Designer 没有)**:

| 改动 | 原因 |
|---|---|
| runner 显式传入 `node_kwargs`(**未改 `graph.py`**) | **上游缺陷**:`__init__` 把 `node_kwargs=None` 补成 `[{} …]`,但下一行构造 `all_node_config_groups` 用的是**原始参数**(仍是 None),默认值当场作废 → `init_nodes()` 里 `zip(agent_names, None)` 抛 `TypeError`。作者的 `run_mmlu.py` 总是显式传字典所以没暴露,`run_gsm8k.py` 传 None 会同样崩 —— 即**除 DirectAnswer 外所有模式在上游都跑不起来** |
| 新增 `CARD/config/shared/{math,code,qa}.json` | CARD 的方法本体是在**组合**上搜索:`node_kwargs` 是「组合名 → 每节点配置(llm_name/role/外部工具)」的字典,Graph 按组合缓存特征供 GCN 打分。配置**逐字取自作者同域文件**(`math/qwen-72B.json` 等),只把 `llm_name` 换成本地模型。作者同时提供异构池(`qwen-7b+qwen-14b+qwen-72b.json`)与单模型配置(`qwen-72B.json`/`gpt-4o.json`),故**单 backbone 是作者支持的配法** |
| 按 domain 解析 agent 类与决策头 | 作者三个 runner 各不相同(run_gsm8k: MathSolver/FinalRefer;run_humaneval: CodeWriting/FinalWriteCode;run_mmlu: AnalyzeAgent/FinalRefer),而 `run_shared.py` 派生自 gsm8k 版,把 gsm8k 的默认值带到了所有 domain。`AnalyzeAgent` 也不能通用 —— 它调 `prompt_set.get_analyze_constraint()`,只有 MMLU 的 prompt set 有 |
| 按 `__init__` 签名过滤 node kwargs | `init_nodes()` 把 node 配置的每个键直接塞进 agent 构造函数,而作者配置里的 `external_tool` / `_type` / `_source` 只有 `AnalyzeAgent`、`CodeWriting` 声明,`MathSolver` 会 `TypeError`。过滤保留作者配置原样,也保住 role 分配(方法真正搜索的那一维) |
| `arun(..., fixed_group=...)` 显式指名组合 | `allow_random_combination=False` 时 `arun` 断言调用方必须指定组合名 |
| `.env` 增加 `SERVER`;`llm_registry` 增 `local` provider;`llm/__init__.py` 的 `together` import 改为可选 | `SERVER` 未设时 `MODEL_NAME_MAP.get(None)` 返回 None,查表前就 `AttributeError`;`qwen3-8b` 不在任何 provider 表里会被译成 `None` 发出去;`together` SDK 未安装(为一个从不调用的托管 API 装 ~90MB,而 `/home/users` 使用率 99%)|

顺带记录一个不改的上游 bug:`llm_registry.get()` 先 `if MY_SERVER == "together": …`,紧接着一个独立的 `if/else` 的 else 分支无条件把选择覆盖成 `GPTChat`,所以 `SERVER=together` 时 TogetherChat 路径是死代码。我们不走这条路,改它就是改作者逻辑。

**注意**:CARD 是 G-Designer 的 fork 但经 black 格式化,`run_gsm8k.py` 与 G-Designer 不逐字相同(`import a, b` vs `import a,b`、`Graph(\n domain=` vs `Graph(domain=`、`x == y` vs `x==y`)。派生器改用空白符容错正则,并对每个锚点做存在性校验 —— 锚点漂移会**大声失败**,而非生成一个"训练不到任何东西"的 runner。

### 3.6 FlowBank `dde948d`

**仓库本体已验证可精确复现**:`inference.py --all --assert` 最大偏差 5.96e-08(容差 1e-4),四个自带 selector 检查点(DROP 0.8349 / MATH 0.6201 / AMC 0.4632 / MBPP 0.7419,均为其自身划分)。**其 5 个数据集中没有 mmlu_pro 的检查点。**

但如 §2 第 5 条,自带 portfolio 与 selector **不可复用**,必须三阶段重跑,因此新增 `shims/diverseflow/`(阶段 1):

| 文件 | 性质 |
|---|---|
| `DiverseFlow/benchmarks/shared_benchmarks.py` | 新增,5 个共享 benchmark 类 |
| `DiverseFlow/run.py` / `run_test.py` | 各追加 5 条 `EXPERIMENT_CONFIGS`(`question_type` 与算子集**逐字取自作者同名数据集的条目**,不自拟) |
| `DiverseFlow/scripts/evaluator.py` | `dataset_configs.update(...)` + `DatasetType` 扩充 |
| `DiverseFlow/config/{config,config2}.yaml` | 新增,指向代理(仅 `.example` 随仓库发布) |
| `datasets/shared_*.jsonl` | 新增 10 个文件(`SHARED_` 前缀避免覆盖作者自带的 `math_test.jsonl` 等) |
| `DiverseFlow/workspace/SHARED_*/` | 从作者同名工作区播种 |

**比 AFlow 省事的一点**:DiverseFlow 原生带齐五个数据集的 benchmark 类、算子集与工作区模板,**MMLU-Pro 也有**,无需任何借用替代。

**绕开 LLM 裁判**:DiverseFlow 的 `evaluate_problem` 在判分前先调 `llm_extract_answer`(每题一次额外 LLM 调用),其 MMLU-Pro benchmark 还上报 `llm_judge_answer` 得出的 `judge_score`。其余六个方法都没有这两项,保留会让 FlowBank 按更宽松的尺度判分,并且额外调用会污染 cost 记账。共享类因此绕开两者。**后果:本表中 FlowBank 的数字与其仓库自报数字不可直接比较**,这是有意的。

**AMC 种子工作流在上游是坏的**:`workspace/AMC/workflows/round_1/graph.py` import 的是 `workspace_qwen3.MATH...` —— 一个仓库中不存在的包名,且指向 MATH 而非 AMC(与 FlowBank 作者用 Qwen3 跑过、但未打包该目录树吻合)。播种时的 import 重写起初只匹配 `workspace.<原名>.` 前缀,漏掉了它,**而 `--check` 也只查这一种前缀,照样报 OK**。现在按通用前缀重写,并断言种子图**只 import 自己的算子**。

**闭环实测**(SHARED_MATH,256 题验证集,3 轮):round 1 = 0.7148、round 2 = 0.6602;5 个 Traceback 全部来自 `operators.py:185 run_code`,即 Programmer 算子执行模型生成 Python 时的报错,属算子正常行为。`--diversity_start_round` 默认 **7**,故正式运行的 `max_rounds` 必须显著大于 7(README 用 30)。

**尚未验证**:阶段 2(CuraFlow 组合筛选,离线)与阶段 3 在**我们数据上**的 selector 训练。阶段 3 的推理代码路径已由 `inference.py` 的精确复现覆盖。

### 3.7 MasRouter

见 §3.1b。

---

## 4. 已知副作用与遗留问题

1. **`daao/tools/schemas/GPTvGenerator.yml` 会在导入仓库时被自动回写**(MetaGPT 式 tool-schema 缓存重写 docstring)。非人为改动,每次运行 DAAO 都会重新出现。建议在最终归档前 `git checkout` 该文件,或加入忽略清单。
2. ~~**代理的上下文裁剪在 prompt ≥ 约 25k token 时仍返回 502**,原因未查明。~~ —— **已查明并修复。**

   两个原因叠加:(a) vLLM 报的是 `at least N`,一个**下界**,而低估可以很严重 —— 同一个 AFlow 优化器 prompt 先报 `25032`、后报 `32095`,差 7000 token,所以按下界算余量推不动收敛;(b) 裁剪重试写成 `continue` 到下一个上游,于是实际可尝试的裁剪次数被 `MAX_RETRIES × 上游数 = 8` 限住,`clamps_left = 12` 从来不是真正的约束,需要约 9 次收缩的 prompt 先把请求次数用光,400 就以 502 冒到调用方。

   修法:裁剪改为在**同一上游内独立循环**(不再受外层重试结构限制),步长从 0.8 收紧到 **0.5**(保证 ≤5 步从 8192 收敛),margin 64 → **256**(吸收下界低估)。实测 prompt 逼近 32k(30500 / 31800 / 32300 词)全部成功,`/stats.context_clamped` 计数可见。

   **这个 bug 当时正在实际杀 AFlow 的轮次**,并且一度被我误当成"8B 写不出可执行代码"的证据。
3. **`shared/models/all-MiniLM-L6-v2` 已裁剪至 175 MB**(删除 onnx / openvino / rust / tf 格式),因 `/home/users` 为全实验室共享且使用率 99%。
4. **temp=0 不保证逐位可复现**:vLLM 连续批处理使同一 prompt 在不同批次下可得不同输出。实测 n=20 时重复间波动 ±5 分,按 √n 推算全量约 ±0.4 分。**因此每个方法需跑 ≥3 次并报 mean ± std**,不可用单次结果宣称小于 2 分的差距。

5. **小模型优化器生成的图会以两种方式失效,两种都必须计量**:(a) 抛异常 —— 如 `operator.ScEnsemble()` 漏参,由重试预算处理并记入 `AFLOW_GRAPH_FAILURES`;(b) **不抛异常但返回错类型** —— 如返回未解包的 `{"response": ...}` 字典,由 §1.2 的解包规则处理并记入 `unwrapped_envelope`。(b) 更危险:它在结果表里长得就像「答全错」,且因为不报错,重试永远不触发。
6. **`--check` 只查标记字符串是不够的**(本轮两次踩到):AFlow 的 `run.py` 带着 SyntaxError 报 OK;DiverseFlow 的 SHARED_AMC import 着别的数据集的算子报 OK。现已分别加上 `ast.parse` 校验与「只 import 自己算子」的断言。其余三个安装器所改文件均已在冒烟中真实执行过(即确实能编译),但**重装时的同类盲点尚未统一封堵**。
7. **补丁脚本自身的两类顺序错误**(已修,记录以免重犯):(a) 用「名字是否出现在文本中」判断某定义是否已注入 —— 而注入的代码块本身就引用该名字,导致定义从未被添加;(b) 后置步骤重新生成文件,把前置补丁覆盖掉。两者都表现为**补丁报告成功、运行时缺失**。
8. **`--max_batches` 只限 MasRouter 的训练循环**,其测试循环仍会跑完整个测试集。正式 sweep 中这是期望行为,冒烟时需手动终止。

---

## 5. 共享层依赖

`shared/requirements-scorer.txt`(14 行):`regex`、`tqdm`、`sympy`、`antlr4-python3-runtime==4.11`、`tenacity`、`aiofiles`、`pandas`、`pydantic`、`loguru`、`tree-sitter`、`tree-sitter-python`、`setuptools<81`。

`setuptools<81` 为必需:新版已移除 `pkg_resources`,而 G-Designer 系依赖的 `class_registry` 仍在使用该废弃 API。

各家族环境:`envs/maas`(maas + daao)、`envs/gdesigner`(gdesigner + card)、`envs/pyg`(flowbank,CPU torch + PyG)、`envs/vllm`(serving)、`envs/tools`(工具链)。

---

## 6. 数据集划分变更(2026-08-24)

为与 FlowBank/AFlow 的发表数字直接可比,math 与 amc 两列改用其官方仓库自带划分;其余三列维持我们的上游权威划分。旧产物按数据集定向归档(archive/math500_products/),不可与新划分产物混表。

1. **math → 官方 L5 划分**(hendrycks MATH test 中 level==5 ∩ 4 科目:Prealgebra / Number Theory / Precalculus / Counting & Probability)。`math_search.jsonl`=119(其 validate)、`math.jsonl`=486(其 test),内容级核验完全一致(486/486、119/119),其 validate∩test=0。替换 MATH-500:500 题混合难度使 8B 接近天花板(daao 0.838),方法间差距被压缩。gold roundtrip:测试集 2,430 次回灌 0 判错、搜索集 595 次 0 判错。gdesigner 系 train_then_eval=116 训(29 整批)+486 评,2 条搜索项落入不完整末批被裁弃。旧集归档 archive/math500_split/。
2. **amc → FlowBank 仓库自带划分,原样采用**:`amc.jsonl`=655(其 amc_test)、`amc_search.jsonl`=165(其 amc_validate)。gold roundtrip:3,275+825 次回灌 0 判错。其全部瑕疵**原样继承并申报而非修复**(其发表数字带同样瑕疵,修复反而破坏可比性):test 内部内容重复 5、validate∩test 内容泄漏 7、纯字母 gold 2(题面未附选项);来源上游无文档(历年 AMC8/10/12 题,经查不含 2022–23 年);难度实测显著低于被替换集(Qwen3-8B 单发 0.700 vs 0.550,各 n=40)。被替换的 AI-MO/aimo-validation-amc(83 题,2022–23 AMC12 全卷)可由 fetch_canonical_data.py 中保留的 build_amc 确定性重建;原文件因备份判断条件写错被覆盖未留档(功能无损,如实记录)。"AMC 借 math 搜索集"的协议替代随之废止,三处硬编码已清(aflow/maas/diverseflow 安装器映射、gdesigner 安装器特例、masrouter loader 回退)。
3. **与 FlowBank 在其余三列上的划分差异(核验记录;这三列维持我们的划分)**:
   - **drop**:我们 test=官方 dev(≈9.5k)固定种子均匀抽 1000(内部同 passage+同问题重复 13,系上游自带)、search=官方 train 抽 256。FlowBank validate 200 + test 800:与我们的 dev 抽样在题目级**零重叠**——若同池抽样期望重叠≈84,零重叠强烈提示其取自官方 train 池;其 test 内部重复 8、validate∩test 泄漏 5。
   - **mmlu_pro**:我们 test=官方 test 14 类 × 80 = 1120、search=扣除 test 后余量分层 252(按 question_id 不相交)。FlowBank 只保留 **13 类(整类删除 math)**,validate 20/类 + test 80/类 = 260 + 1040;其 validate∩test 泄漏 1、test 内部重复 4。连带核出上游缺陷:MMLU-Pro test 存在同题不同 question_id 的近重复,使我们 search 与 test 内容级重合 3 条(0.27%),id 级仍不相交,如实申报不改。
   - **mbpp**:我们 test=官方测试段全量 500(task_id 11–510)、search=官方 train 段抽 256;FlowBank 86+341,其 test 仅约 60% 落在官方测试段。
4. **常量与守卫**:DATASET_COST math 500→486、amc 83→655;TRAIN_BATCHES math 64→29、amc 64→41(165 裁到 164 整批)。fetch_canonical_data.py / fetch_search_data.py 中被取代的构建器已停用,防止误重建覆盖采用文件。
5. **amc 泄漏核验补充(2026-08-24)**:除 §6.2 已申报的 validate∩test 7 条外,跨列内容级核验:math 搜索(119)∩ amc 测试(655)= 0(关键方向干净);amc 搜索(165)∩ math 测试(486)= 1(math 列上界 0.2 分,对九方法均匀);math 测试 ∩ amc 测试 = 4(两列均为评测侧,不构成训练泄漏,仅两列不完全独立)。历年 AMC 真题全网公开,预训练级污染对所有公开数学基准同等存在,无法消除,如实声明。
6. **amc 测试集泄漏剔除(用户决定,2026-08-24)**:对 §6.2 采用方式的修正——测试集不得包含搜索阶段可见的题,故将内容出现在 amc_validate 中的 7 条测试行剔除(含其在 test 内的全部副本),655 → **648**,剔除行存档 archive/amc_leak_removed.jsonl,manifest 同步(sha256=26c80bcc1d54f9b8)。剔除后 gold roundtrip 3,240 次回灌 0 判错。与 FlowBank 发表数字对表时需注明口径差 7 题(≈1.1%,我方更严)。DATASET_COST amc 655→648。
7. **出处逐字节核验(2026-08-24,AFlow 官方数据包 aflow_data.tar.gz,2024-10 打包,自 Google Drive 下载,留档 archive/aflow_official_bundle/)**:包内仅六基准(humaneval/gsm8k/hotpotqa/math/drop/mbpp),**无 amc、无 mmlu_pro**。FlowBank 的 math/drop/mbpp 七个文件与包内同名文件 md5 全部一致(math_validate=3cb6f2dd…、math_test=03755669…、drop_validate=3938d0fa…、drop_test=b23f9a84…、mbpp_validate=367fabdb…、mbpp_test=3e0cb353…、mbpp_public_test=5a34f584…)——即 FlowBank 的这三列原封继承 AFlow;其 amc(含 §6.5-6.6 的泄漏/重复)与删掉 math 类的 mmlu_pro 均为 FlowBank 自制。此前对 drop 取自 train 池、mbpp 40% 出官方测试段的判定,构造责任归 AFlow,FlowBank 沿用。
8. **AFlow 官方 mbpp/drop 文件的缺陷精确计量(2026-08-24,基于留档数据包)**:mbpp test 341 = 官方 test 段 203(59.5%)+ train 段 97 + validation 段 35 + few-shot 提示区 6(task_id 1-10,社区通用示例题,污染最重);validate 86 同样混切(54+23+8+1);但其 mbpp 内部干净:validate∩test 泄漏 0、test 重复 0。drop 的内部缺陷见 §6.3(validate∩test 泄漏 5、test 重复 8、疑取自官方 train 池)。两列我们维持自建规范划分,不受影响;若日后需与其发表数字对表,按 amc 先例:原样采用 + 泄漏剔除(drop 需剔,mbpp 无需)+ 申报。
9. **更正(2026-08-24,取代 §6.3 与 §6.8 中的两处错误判断)**:此前AFlow drop 疑取自官方 train 池与最初的与我们零重叠均为**本方比对缺陷**——其 context 尾部带 \nAnswer: 后缀,未剥离导致指纹全错。剥离后直接核验官方原始发布(ai2 drop_dataset.zip):其 1000 条 passage 与 question 均逐字命中官方 **dev**(1000/1000,含官方自带的排版错字),即与我们同池(dev)不同抽样。其内部 validate∩test 泄漏与 test 重复维持原判。mbpp 补充:以 code 字段验得 task_id 为官方真实 id(350/427 与官方 code 一致),故跨官方四区混切成立;另发现 **201/427 行的题面措辞与官方同 id 文本不一致**(轻度改写)、77/427 行 code 有改动——改写未见上游声明。我方三列维持自建划分的结论不变;我方自身上游近重复一并申报:mmlu_pro 搜索∩测试内容级 3 条、mbpp 2 条、drop 测试集内上游自带重复 13 条。
   §6.9 数字补充:整题级(passage+question)核验——我们 drop 测试 1000 与其 test 重叠 104、与其 validate 重叠 28(同池均匀抽样期望约 84/21,量级吻合,进一步佐证双方同取自官方 dev);其内部 validate∩test 整题泄漏 5。
10. **math/amc 判分第三层重试:答案装饰剥离(2026-08-24)**:线上 L5 数据的 correct-but-zero 审计抓到两类做对却 0 分——gold 28\% vs 模型 28(尾部单位)、gold -2+\sqrt{3} vs 模型 a = -2+\sqrt{3}(头部变量赋值)。在既有重试永不降分架构上新增第三层:两侧对称剥离尾部 \%/%/^\circ/°/degrees 与头部单字母变量赋值前缀,仅在前两层判 0 后触发,命中计数 recovered_answer_dressing;28\% vs 0.28 剥后仍不相等,不会误给分。gold roundtrip 修后复验全对(math 2,430、amc 3,240,0 判错);复跑审计确认该类清零,残余案例均为模型真错(符号/分量/多答案)。生效边界:正式 math 作业 01:52 起跑、修复 03:4x 部署,九个在跑进程内存中仍为旧判分,即搜索期最初约 2 小时的奖励信号未含此层(对九方法均匀);最终表分数全部由 collect 以当前判分器从落盘预测重判,不受影响。
11. **遗留申报事项汇总(2026-08-24 补录)**:(a) masrouter 环境缺 wikipedia 包,WikiSearcher 角色以报错文本参与对话——作者依赖未声明,对所有数据集均匀,不修,如实计入;(b) masrouter 作者代码按 int(N/batch) 丢弃末尾不完整批,mmlu_pro 实跑 992/1000 等,收集时按实际判分条数报告;(c) mbpp 有 1 题参考解为 class Node 结构,被作者管线的函数式 sanitize 丢弃,该题对全部方法计 0,均匀;(d) DROP F1 的词形差异(单复数等)为官方指标固有行为,不做词干化,残余做对却 0 分率每格 0.2-0.4% 且均匀。
12. **G-Designer 家族生成代码执行缺口修复(2026-08-24)**:§4 修复 2 的 30s 硬超时当时只覆盖了 maas 系/aflow(flowbank 自身安全),漏掉了 G-Designer 家族 math 域独有的裸 exec 路径(tools/coding/python_executor.py 的 execute_code_get_return,仅 MathSolver 调用,故 drop/mmlu_pro 从未触发)。官方 L5 数据下模型生成的 compute_n_plus_k 永久阻塞(0 CPU),看门狗 45 分钟双静默告警 + SIGUSR1 栈转储定位。修法:该路径改走仓库自有的 function_with_timeout(线程 join 超时,30s,与其余家族策略一致),安装器新增 patch_exec_timeout 步骤(标记幂等 + ast 校验 + --check 核验),gdesigner 与 card 两仓库同时生效。代价:gdesigner_authordefault/math 作业损失 2.7 小时重跑;其余三个在跑的家族 math 作业内存中仍为未修版本,由看门狗盯守,若复发则杀掉重跑即自动带上补丁。
   §6.12 续(2026-08-24 04:5x 决定):gdesigner/card/card_authordefault 三个 math 作业实测已完成 83%/100%/65%,不重跑、带旧版执行策略跑完,理由与证据:(a) 该缺陷只致卡死不致错分,跑完即有效;(b) 三格请求间隔扫描(2,567-3,623 次请求)最大间隔 299-302s,封顶于客户端长生成上限,不存在超过 5 分钟的执行阻塞;>30s 间隔数(42-59)与训练批次边界数(29)同量级,主体归因本地训练步;(c) 残余不对称上界:若存在 30-300s 的慢代码题,旧版会等出结果、新版(gdesigner_authordefault 所用)记错误——影响至多个位数题目,如实申报。
13. **collect 重判行低规格缺陷(2026-08-24 发现并修复)**:regrade()/from_maas_csv 用 {ref_text, answer, code} 存根行重判,mmlu_pro 缺 options、mbpp 缺可执行的 test 行——前者在 regrade() 中被兜底 except 吞掉并静默返回落盘旧判分(备注仍称 recomputed),后者在首个 mbpp 格完成时当场崩溃暴露(daao/mbpp KeyError)。修复:新增 _grading_row(),mmlu_pro 给满字母空间(与审计工具一致),mbpp 按参考解回连数据集行(划分内唯一),连不上计数申报。修复后全量真重判核验:gdesigner 系 drop/mmlu_pro 数字与此前静默回退值逐位一致(该两列判分器从未变更,回退恰好无害),最终表无需更正;daao/mmlu_pro 首次获得真实重判分(此前一直因崩溃标 pending)。
14. **mmlu_pro 字母抽取器缺右边界守卫(2026-08-24,用户质疑 daao 低分引出)**:answer_lead 模式的捕获组后无 (?![A-Za-z]),对 "### Final Answer:\nAnswer: (C)" 形回复,第一个 Answer: 的先导吃掉换行后把第二个单词 Answer 的首字母 A 捕获为答案,真正的 (C) 永不可达。daao/mmlu_pro 1120 题中 168 题末尾明确写对字母却被判错(0.5196→修后 0.6562);单字母 gold 恰好躲过 correct-but-zero 审计的子串预筛(len>2 条件),故此前未暴露。修复:answer_lead/option_word 捕获组加 (?![A-Za-z]);杀手形状冻结进 gold roundtrip(mmlu_pro 6,720 次回灌 0 判错)。影响面:已完成格由 collect 重判自动更正;搜索期奖励信号的低估对九方法均匀(同一抽取器);masrouter 无逐题落盘、终分为运行时判定,故为其 shim 增加纯日志性的逐题落盘(scored_items_<ds>.jsonl,uid 命名空间区分训练/评测),collect 优先从落盘重判,其 mmlu_pro 作业于 09:02 以修复版判分器重启。
15. **全方法全层"做对却判0"清查 + 搜索期奖励失真定量(2026-08-24,用户要求)**:(a) 评测侧:mmlu_pro 修复后对全部有逐题记录的方法(gdesigner 系×4、daao)复筛,字母残余=0,"选项原文"初筛命中经人审全为模型真错;drop/math/amc 全层审计,除已申报的 DROP 词形残余(0.2-0.4%)外,新增修复 math 金标准中 LaTeX 转义美元符 \\$ 未被排版归一化剥除(gold \\$36 vs 模型 36,gdesigner 系每格约 1 题),修后 roundtrip 保持全对;其余嫌疑逐条人审均为真错(符号/数值/多解/词形)。(b) 搜索侧:aflow/mmlu_pro 全部 21 轮以修复版重判——早期 4 轮被压低 5-8 分,但最优轮不变(round_18);flowbank/mmlu_pro 27 轮中 6 轮被压低 3-4 分,最优轮不变(round_14);daao/mmlu_pro 单轮搜索,奖励整体压低 11.5 分(水平位移,架构级选择在 VAE 权重内不可事后审计,按同轮内近似均匀申报)。结论:两个按轮选择的方法零选择损失;既定流程(aflow_test 选轮前 regrade_rounds、flowbank_pipeline 重评)对任何残余选择偏差有兜底;早期轮误导奖励对"候选生成"的反事实影响不可测,但缺陷惩罚的是输出格式而非能力,属二阶效应,对同期搜索的全部方法同源同向。
16. **全矩阵"做对却判0"终审(2026-08-24,用户令,约4万条记录,搜索+测试层全覆盖)**:mmlu_pro(21,224 条,含 aflow/flowbank 全部搜索轮)修复版预筛下嫌疑=0;mbpp(4,000+ 条)嫌疑=0;masrouter 新落盘 528 条嫌疑=0;drop 嫌疑均为已申报的官方 F1 词形类(males/females、Zürich/Zürichers 等);math/amc 嫌疑逐条人审均为模型真错(倍数、符号、多解、附加项),$ 前缀类修后清零(daao/math 重判 0.7984→0.8004)。重判-落盘差列如实记录各修复的回收量(daao/mmlu_pro +0.137 最大)。另申报:极少数存储回复为代理 502 错误文本的样本按 0 分计(基础设施失败,占比 ~0.2%,对各方法均匀)。结论:不触发任何重跑——最终分层零残余误判(重判自动更正);搜索层可测部分选择零变化(aflow round_18、flowbank round_14 不变),不可测部分为九方法同源同向的格式压力,重跑单方法反破坏均匀性,全量重跑无可证明收益。
17. **终审:全层"做对却判0"清查完成(2026-08-24,约2.1万条判0记录逐条二次意见+人工裁决)**:(a) 根因级发现——AFlow amc.py 的 symbolic_equal 在解析前删除全部反斜杠(s.replace(chr(92),"")),其符号等价自始至终从未生效,受害面为 amc 列全程(\frac38 vs \frac{3}{8} 等 67 条存量记录);修复为 bench 第四层重试:未被篡改的 parse_latex 直接等价 + 数值容差,并防前缀解析陷阱(含逗号/分号/下标/关系符的输入不走该层,防 (4,24)=(4,12)、101_3=101_2 类误授);单侧进位制后缀(-221_3 vs -221)精确处理。(b) drop 抽取三修:token 级剥马克当emphasis(**4.4%** 曾被剥成 44)、token 级百分号数字识别(长 span 内嵌 4.4% 同病)、缩写句点不作句尾(RB T.J. Duckett... 曾在 T.J 截断致 Kevin Jones 未参与比较);gold roundtrip 9,948 次保持全对。(c) 终值:mmlu_pro 判0 9,982 条中冤枉 0;amc 2,886 条中 1(gold 为未求值和式 \sum binom^3 而模型答其值 2252,系 FlowBank 金标准书写风格,已在防陷阱守卫的已知代价内,如实申报不改);math 2,916 条中 0(3 条嫌疑均为模型真错);drop 5,241 条中 0(352 条散文提及/答非所问判0正确,9 条 Answer 行案例逐条人审均为真错或已申报词形类)。(d) 选轮稳定性(修复判分器下重验):aflow/mmlu_pro、flowbank/mmlu_pro、flowbank/amc 最优轮均不变。合计残余率 1/21,025≈0.005%。
18. **mmlu_pro 全列重跑(用户决定,2026-08-24 17:45)**:为使该列搜索奖励与最终判分同尺(字母抽取修复前后统一),8 方法从零重跑(旧产物归档 archive/mmlupro_rescore_1745/),masrouter 保留其 09:02 起的修复版判分器干净跑。夜间部署 nanny.sh 自愈(失败且日志静默>15分的格自动重跑,每格≤2次、总额≤10、代理健在才动手,全程留痕 logs/nanny.log)。
19. **amc 数据文件的顺序结构与"对照曲线"裁决法(2026-08-24 夜)**:FlowBank amc_test 按赛段分组、压轴难题集中在文件尾部——daao 干净跑的分段零分率(31/19/12/6/13/61%)证实尾段 61% 零分为数据固有。据此:card_authordefault/amc 的尾段 66% 零分判为数据顺序效应而非污染,曾误令重跑,已撤销并恢复原有效记录(0.6564),完工挂死按先例改判 ok;card/amc 的中段零分洪峰(91/82%,对照同段仅 19/12%)与数据曲线不符,风暴污染结论维持、重跑继续。另修复 aflow benchmarks/mbpp.py 测试线程非 daemon 导致的泄漏挂死(与 gdesigner 家族同类,装载器 patch_mbpp_daemon_thread),aflow/mbpp 自 13 轮断点续跑。17-21 点无第二场风暴(全窗失败共约 117 次,系正常波动)。
20. **判分第五、六层:无序裸逗号列表 + 向量记法(2026-08-25 晨)**:整夜增量误判扫描在 gdesigner_authordefault/math 抓到真冤案——gold `\frac{3}{4}, -\frac{3}{4}` vs 模型 `-\frac{3}{4}, \frac{3}{4}`,同一解集仅顺序颠倒;作者判分整串比较、第四层"含逗号即拒",均看不见该等价。随后全库扫描(v5 以来全部 math+amc 记录,logs/blast_comma_multiset.log、blast_math_show60.log)抓到第二类:gold `(7,21,35)` vs `\begin{pmatrix}7\\21\\35\end{pmatrix}`(daao/math 测试集 1 条,值与顺序全对、仅记法不同)。修复:第五层 `_comma_multiset_equal`(仅当两侧均为**不被括号整体包住**的裸逗号列表、2–5 项且等长,元素两两(作者判分 ∨ sympy 直连)做完美匹配;比较用 relax 形而非 dressing 形——dressing 会剥坐标对外括号,在该层比较会把 `(-7,10)` vs `(10,-7)` 误判相等);第六层 `_vector_tuple_equal`([pb]matrix 整串向量 vs 圆括号元组,**严格按位**,排除 vmatrix 因其为行列式)。护栏初版"末字符为 `}` 即算被包住"被回归当场打脸(`\frac{3}{4}` 天然以 `}` 结尾拦住正主),改为括号配平判定 `_enclosed_by_brackets`。回归 `audits/regression_comma_multiset.py` 19 用例(2 翻正 + 17 守,含坐标/区间/比例/向量翻转与 vmatrix)全过;math/amc gold roundtrip 2,430/3,240 条保持全 1.0。**影响面(全库)**:math 51 + amc 4 = 55 条嫌疑逐条人工裁决,真冤案两类:排列类约 5 条记录(全在 gdesigner_authordefault/math 的 train+eval 流量,同题多副本表明属训练批次,占该格 0.35%,仅影响训练奖励聚合;该格重判均分 0.6999→0.7025);向量类 1 条(daao/math 测试集,collect 重判自愈,+1/486≈0.002)。**搜索阶段(aflow/flowbank)55 条中的 22 条嫌疑全为真错——搜索零冤案**;amc 全列零真冤案。在跑作业(maas/math、masrouter/math、card/amc、masrouter/amc)进程内存仍为第五层前判分器:其测试分由 collect 落盘重判自愈,搜索奖励按 amc 第四层先例作"判分器版本轻微不齐"申报。确认扫描 logs/blast_confirm_tier6.log。

    **更正(确认扫描后)**:上文"搜索阶段零冤案"仅对**嫌疑名单**成立——子串预筛看不见排列/记法差异。确认扫描(logs/blast_confirm_tier6.log)的重判满分增量暴露了预筛盲区:第六层在搜索阶段追认 aflow/math 约 10 条、flowbank/math 约 6 条正确记录(各占该格 0.49%/0.28%),各 gdesigner/card 系格另有 +2~6 条(train+eval 混合流量)。math 全库第五+六层合计追认约 35 条(全库 8,482 条 math 记录、约 3,050 条判 0 之中);amc 零追认(确认扫描满分数逐格与前次一致)。搜索期已下发的奖励不可追改,占比不足以改变优化轨迹(单轮均分摄动 ≤0.006),照例申报;aflow 轮次选择不受影响——aflow_test 前强制 regrade_rounds 以当前判分器重判(既定流程)。

21. **杀漏进程连锁清理与孤儿接生(2026-08-24 21:0x–22:1x UTC)**:21:10 击杀误发的 card_authordefault/amc 重跑时只杀了表层——其 driver(1499972)与 21:09:32 已拉起的 search 子进程(1499975)存活,后者跑了 56 分钟、向该格 dump 追加约 96 条新命名空间搜索记录(eval 产物与 0.6564 有效分未受影响,WHY 文件已追记);22:1x 补杀时连带规则(父进程含 sweep.py 即杀)误杀共享 amc 列 driver(624918),maas/amc 与 masrouter/amc 两个健康 job 成孤儿(进程仍在正常干活)——已部署 shepherd_amc.sh:maas/amc 退出后自动 relaunch_one 补记账(checkpoint 续跑已验证安全);masrouter/amc 仅记录、待人工按产物裁决(无断点,盲拉会从零重训)。同批清理:11 小时前"完工挂死"的旧 card_authordefault/amc 进程(880008,当时同样杀漏)、8 个 flowbank/mbpp 完工挂死僵尸(DiverseFlow 执行线程非 daemon,同 §6.19 类;该格 status=ok 产物完整)、2 个重复 watchdog、2 个挂死约 12 小时的旧审计进程、1 个挂 13 小时的 ssh 启动壳。另:aflow/mbpp 以 daemon 补丁于 21:12 断点续跑、21:23 正常完工(status ok),mbpp 列仅剩 masrouter 在跑。教训:(a) ssh 后台链须用重定向子壳 `( ... ) </dev/null >/dev/null 2>&1` 包裹,否则挂住通道且远端留常驻壳;(b) 杀一个格要对 driver+job 全树逐 pid 核对,连带规则禁用 `*sweep.py*` 这类会命中共享 driver 的宽模式。

22. **masrouter/mbpp 记 DNF(2026-08-28,用户拍板)**:四次尝试均未完成(22h/3.5h 死因随日志轮换丢失;第 3 次被 driver 重启接管;第 4 次干净跑满 24h 被 sweep 超时杀于 Epoch 2/Batch 2,约 18/80 批)。根因:mbpp 上单批(16 样本)成本 1–3 小时(多智能体工作流 × 代码生成+执行),全程需约 4–5 天;作者代码存 per-epoch 检查点但从不加载,无断点能力。对排名零影响:masrouter 四列均分 0.693 列第五,即使 mbpp 取 0.70,五列均分 0.694 仍第五。终表该格标 DNF。

23. **缩进代码被 sanitize 静默丢弃(2026-08-29 发现并修复)**:上游 `sanitize(code, entrypoint)` 只保留与 entry_point 同名的**顶层**定义。Qwen3-8B 在改写既有解法时高频返回整体缩进两格的函数(SelfRefine 路径尤甚),此时 `def` 不在顶层,sanitize 返回空串——模型写对的代码被丢弃、该题记 0 分,且**全程无任何报错**,CSV 的 prediction 列为空但 logprob/vae 正常写入,日志层面完全不可见。实测(本轮 sweep 全部 mbpp 代码回复 17,141 条):**修前空串率 31.3%(5,364 条),修后 1.1%(194 条),救回 5,170 条**;daao/mbpp 测试集 142 个空预测中,**138 个的模型回复确实定义了所要求的函数名**。修复:在 `code_fill` 调 sanitize 前对整段回复做公共缩进剥离(仅当首个 def/class 带缩进时触发,dedent 后必须仍有顶层定义才采用,否则原样返回),不改变代码语义与相对结构。端到端验证:缩进形状恢复满额、顶层形状逐字节不变、函数名不匹配仍正确返回空。已固化为 maas-family 安装器步骤 + `--check` 断言。

**第 23 条的因果补充(2026-08-29 查明,2026-08-30 完成修复)**:缩进不是模型的随机习惯,**是我们自己的格式示例诱导的**。旧版 `shared/bench.py` 的 mbpp 格式要求里,示例本身带两格缩进(`  def example_name(x):` / `      return x`),模型照抄了这个缩进。证据:11,465 条含代码的 Generate 回复里,缩进者 11,083 条中 **11,005 条(99.3%)恰好两个空格**,与旧示例精确一致。最终处理分两层:(a)示例改为顶格 `def example_name(x):`,直接消除诱因;(b)MaAS/DAAO、AFlow、FlowBank 及共享 MBPP 判分入口统一做保守公共缩进剥离,防模型偶发整体缩进。G-Designer/CARD 不走该 sanitize 路径,其代码执行器保留作者行为。修后 623 次新调用中旧缩进 ICL=0、正确顶格 ICL=620、失败 transcript=0;MBPP 1,000 次 gold 回灌和全数据集 23,338 次回灌均 0 失败;8 个 MBPP 配置的 6 题冒烟全部通过(MasRouter 按决定未跑)。该变更要求 MBPP 正式结果全量重跑,旧 MBPP 列不得混用。

24. **源码迁移与从零重建审计(2026-08-30)**:根项目整理为源码型 Git 仓库,作者仓库不嵌套提交,而由 `upstreams.lock.json` 固定七个完整 commit、`scripts/bootstrap_upstreams.py` 克隆后运行五个 shim 安装器。模型、环境、缓存、日志、结果、checkpoint、archive 全部排除。以干净 commit 从零安装后逐文件与 A800 当前树做 checksum 对比,抓到一处此前仅存在于旧服务器工作树的修复:FlowBank MiniLM 分支后必须使用 `elif random`,否则会落入 OpenAI 分支的 `else`;现已编码进安装器并加入 `--check` 断言。G-Designer/CARD 的悬空视觉导入注释也改为真正幂等。五个安装器连续重复执行前后内容指纹一致,全部 `--check` 0 失败。
