# 文件地图(2026-08-23 整理)

三层结构,按"还会不会被运行"划分。**核心管线在根目录,检查工具在 `audits/`,用完的一次性诊断在 `attic/`。**

## 根目录 —— 实验管线(tmux / 互相调用,不要移动)

| 文件 | 作用 |
|---|---|
| `sweep.py` | 跑 方法×数据集 矩阵;写 `runs_*/`、per-job protocol.json、命名空间 |
| `collect.py` | 汇总分数;**从存下来的预测用当前判分器重判**,不信任运行时写入的分 |
| `vllm_proxy.py` | 全部方法的唯一 LLM 入口;温度 0、presence_penalty 1.0、撞满重跑一次、超限返回空答案;transcripts 成功失败都记 |
| `watchdog.py` | 每 10 分钟:代理健康、卡死检测(日志+命名空间双信号)、污染扫描(调 `audits/live_contamination.py`) |
| `launch_vllm.sh` | 起 vLLM(40960 窗口;显存按自身预算推导) |
| `stop_all_jobs.py` | 安全停作业(/proc 精确匹配,不误杀 ssh) |
| `separate_runs.py` | 换协议前把**全部** 42 个产物路径归档(runs、日志、各仓库工作区) |
| `aflow_test.py` | AFlow 收尾:用当前判分器重判各轮后,选最好轮次跑留出集 |
| `flowbank_pipeline.py` | FlowBank 收尾 2a–3e(唯一分数来源) |
| `make_train_then_eval.py` / `fetch_*.py` | 数据集构建 |
| `shared/bench.py` | **唯一判分权威**;官方 DROP F1、sympy 数学等价(缺 antlr4 时拒绝判分)、XML 信封剥离、MBPP 入口修正 |
| `shared/exec_guard.py` | 生成代码的 30s 可杀执行 |
| `shims/` | 七方法安装器(全部幂等,带 --check) |

## `audits/` —— 活的检查工具(每次改判分/提示词后重跑)

| 文件 | 检查什么 | 何时跑 |
|---|---|---|
| `test_gold_roundtrip.py` | **标准答案回灌必须得满分**(含 XML 信封形状) | 改 bench.py 后必跑 |
| `test_equivalent_forms.py` | 等价写法(14/3 vs \frac{14}{3} vs 0.4667)是否得分 | 同上 |
| `correct_but_zero.py` | 存下来的预测里"做对却 0 分"的比例,全数据集全存储格式 | 每批结果出来后 |
| `live_contamination.py` | 活流量里跨数据集措辞(watchdog 每 10 分钟自动调) | 自动 |
| `stage_audit.py` | 按阶段聚类打印某格子的真实指令行+回复尾部 | 人工抽查 |
| `scorer_reference_check.py` | 我们的判分 vs 独立参照(官方 DROP / sympy)逐条对拍 | 大改后 |
| `regrade_rounds.py` | 各优化轮次重判,看选中的最好轮会不会变 | aflow/flowbank 收尾前 |
| `inspect_io.py` | 按 namespace/grep/errors 查 transcripts(报错永不截断) | 排查任何问题的第一步 |
| `test_proxy_protocol.py` / `test_namespace.py` / `test_exec_guard.py` | 代理采样协议 / 命名空间通路 / 执行防护 | 改对应组件后 |
| `render_prompt_sets.py` / `scan_escape_corruption.py` / `prompt_baseline.py` | gdesigner 提示集渲染 / 转义损坏 / 提示词基线 diff | 改提示词后 |
| `replay_overflows.py` | 重放超限请求验证窗口 | 改窗口后 |

## `attic/` —— 已完成使命的一次性诊断(只留档,不再运行)

复读测量系列(penalty_*/retry_on_loop/degeneracy)、卡死排查系列(find_hanging_solution/inspect_stall_window/test_generated_code_hangs)、污染初查系列(scan_prompt_contamination/prompt_provenance/masrouter_routing/letter_space)、判分排查过程件(audit_drop_extraction/rescore_csv/inspect_regressions/paired_protocol_diff/show_undercredit)、并发测量(concurrency_probe/job_concurrency)等。结论都已写进 REPRODUCTION_CHANGES.md 和 audits/ 里的正式工具。
