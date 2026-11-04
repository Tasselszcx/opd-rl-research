# 哪些文件接入了我们的 idea

新增部分分两层：**外层决定怎么教、跑哪些分支；G-OPD 内层执行一次 warmup 或 RL。** 参数训练、分布式通信、模型加载和 vLLM rollout 继续使用原框架。

## 先看这几个入口

| 你要做的事 | 文件 | 改什么 |
|---|---|---|
| 换学生、教师、预算和教法 | `mvp_configs/smoke_8gpu.json`、`pilot_8gpu.json`、`budget_8gpu.json` | 模型路径、步数/时间配额、动作列表、轮数 |
| 看配置如何校验 | `meta_teaching/config.py` | 默认参数、环境变量展开、未知字段与非法组合检查 |
| 换教学策略或价值模型 | `meta_teaching/policy.py` | `select` 选动作，`predict` 估计收益，`update` 学真实收益 |
| 换教师看到的参考内容 | `verl/verl/trainer/ppo/meta_teaching_hooks.py` | `partial_reference`、`prepare_teacher_inputs` |
| 改调用教师前的筛选规则 | 同上 | `choose_query_indices`，目前为随机或学生平均 surprisal |
| 改真实训练参数 | `meta_teaching/backend.py` | `build_phase_command` 构造 warmup/RL/eval 的 Hydra 参数 |
| 换数据 | `meta_teaching/data.py` | JSONL 到 Parquet、参考信息隔离、split/hash 检查 |
| 换任务奖励 | `mvp_tools/math_reward.py` | 正式数学验证器；演示数字验证器单独命名 |

## 外层新增模块

| 文件 | 在做什么 | 重点边界 |
|---|---|---|
| `meta_teaching/__main__.py` | `plan`、`preflight`、`run`、`simulate`、`status` 命令入口 | 模拟运行与真实训练分开 |
| `meta_teaching/runner.py` | 从同一起点组织分支，收集后续 RL 结果，更新策略，选择下一轮学生 | 不把 warmup 分数当作长期收益，不用 test 更新策略 |
| `meta_teaching/backend.py` | 启动独立 G-OPD 子进程，保存命令、日志和实际运行时间 | RL/eval 不传教师路径；失败阶段不静默重跑 |
| `meta_teaching/simulation.py` | 在不加载模型的情况下检查外层调度 | 分数和成本是明确标记的 fixture，不是性能结果 |
| `meta_teaching/policy.py` | 动作级 ridge 收益估计器和 epsilon-greedy 选择 | 纯 Python；不是 PPO critic，不更新教师模型权重 |
| `meta_teaching/config.py` | 读取 JSON、展开模型环境变量、校验实验配置 | 拼错字段直接报错；禁止无效 batch/预算组合 |
| `meta_teaching/preflight.py` | 检查模型、tokenizer、数据、依赖、GPU | 文件检查不等于权重已加载成功 |
| `meta_teaching/data.py` | 准备 train/meta_val/test 与文件清单 | 参考解只保留在 train metadata；检测重复题和 ID |
| `meta_teaching/common.py` | 原子 JSON 写入、哈希、路径和数字校验 | 中断时尽量保留完整状态文件 |
| `mvp_tools/math_reward.py` | 将回答交给明确指定的数学验证器 | 正式依赖缺失就报错，不自动改用演示评分 |
| `scripts/mvp_environment.sh` | 查环境；在显式命令下安装 CPU 或 Hopper profile | 默认不安装；不停止其他 Ray/训练任务 |
| `mvp_requirements_cpu.txt` | 数据准备和外层测试依赖 | 不安装 torch/vLLM/整套 verl |
| `mvp_requirements_hopper.txt` | 旧 G-OPD CUDA 12 训练栈的直接版本约束 | TensorDict 冲突已统一；Blackwell 应使用另外验证的环境 |
| `mvp_configs/simulation.json` | 不依赖模型环境变量的默认模拟配置 | 用 `simulate` 即可运行 |
| `mvp_configs/smoke_8gpu.json` | 演示数据上的最短 GPU 流程 | 先验证训练和切换阶段 |
| `mvp_configs/pilot_8gpu.json` | 真实数据上的固定步数探索 | 不称作等算力结果 |
| `mvp_configs/budget_8gpu.json` | 真实数据上的固定时间配额实验 | 仍记录实际开销与超时 |

## 对 G-OPD 的修改

### `verl/verl/trainer/main_ppo.py`

在创建训练 worker 前检查实验阶段。warmup 必须启用 OPD；RL/eval 必须关掉 reference worker 和教师路径。

使用同一 tokenizer 时，直接保留学生回答的 token ID，不把回答先解码再用教师 tokenizer 重编码。这样，teacher 给出的 sampled-token 分数与 student 真正生成的 token 对齐。

### `verl/verl/trainer/ppo/meta_teaching_hooks.py`（新增）

这是教学动作进入训练的主要位置：

| 函数/类 | 具体作用 |
|---|---|
| `validate_phase_config` | 检查 warmup/RL/eval 分离，拒绝 LoRA、多轮、额外 KL 等当前未控制的路径 |
| `assert_same_tokenizer` | 检查词表 ID、特殊 token、模板、分词规则 |
| `partial_reference` | 去掉显式最终答案，再按字符截取参考推理前缀 |
| `prepare_teacher_inputs` | 仅为教师构造参考条件，保留学生问题和学生回答 token |
| `choose_query_indices` | 根据学生侧信息选样本，不先调用教师 |
| `score_selected_teacher` | 只把选中的样本发给教师，补齐分布式 batch，评分后写回原位置 |
| `PhaseState` | 记录阶段步数、teacher 调用、最终验证分数、checkpoint 和截止时间 |

未选中的样本使用学生原 log-prob 作为 reference 锚点，在 vanilla sampled-token OPD 中令对应蒸馏 advantage 为零。warmup 关闭额外任务 advantage、熵项和 G-OPD 校正路径，避免未指导样本从这些支路更新。

教师计数区分“真正选中多少样本”和“分布式补齐后实际评分多少样本”。记录的输入 token 包含教师参考信息，而不只是学生回答。

### `verl/verl/trainer/ppo/ray_trainer.py`

把上述 hooks 接到实际训练循环：在学生 rollout 之后、教师评分之前执行选择；warmup 完成时保存完整状态和最终验证结果。

新阶段的步数按“从恢复点再跑多少步”解释。例如恢复到第 100 步，本阶段要求 20 步，就跑到 120 步。使用恒定学习率，避免阶段长度改变 scheduler 语义。

deadline 在 batch 边界检查，结束前预留评测时间。最后分数必须对应最后 checkpoint；不能误用若干步之前的旧验证分数。

### `verl/verl/workers/fsdp_workers.py`

明确设置训练和 rollout 随机种子，并给 rollout RNG 单独保存、恢复文件。原 actor checkpoint 保留模型、optimizer、scheduler 和训练 RNG。

这些文件提高分支可比性，但不会保存所有 vLLM 内部执行状态，不能声称跨进程逐位复现。

### `verl/verl/workers/config/rollout.py`

为 rollout 配置补充显式 seed 字段，使分支种子能传到推理后端。

### `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

把分支 seed 传给 vLLM 初始化，避免所有分支都用其默认值。

## 一次教法如何影响学生

```text
配置 actions
  → 外层策略选择 plain / partial / full
  → backend 把 exposure、query_fraction、gate 写入训练参数
  → hooks 在学生 rollout 后选择需要评分的样本
  → 只对教师加入参考内容
  → 原 G-OPD OPD loss 更新学生
  → 保存 checkpoint，退出 warmup 进程
  → 启动不加载教师的 RL 进程
  → 最终 meta_val 分数与 direct_rl 配对比较
  → policy.update(context, action_id, reward)
```

因此，策略训练标签来自独立 RL 的真实结果；学生的 RL 训练仍然只看任务奖励。

## 测试分别证明什么

| 测试文件 | 覆盖内容 |
|---|---|
| `tests_mvp/test_policy.py` | 实测标签能改变选策、不同上下文可选不同动作、冷启动不固定偏向首动作、保存恢复 RNG、成本单位 |
| `tests_mvp/test_backend_hooks.py` | 教师条件不改学生输入、token 对齐、查询前 gate、未选样本处理、阶段隔离、截止时间和摘要 |
| `tests_mvp/test_data.py` | 三份数据隔离、参考解字段处理、Parquet 和哈希 |
| `tests_mvp/test_preflight.py` | tokenizer 不兼容、缺文件、缺分片和错误数据设置能被发现 |
| `tests_mvp/test_environment_contract.py` | 默认只读、安装要求独立环境、Hopper/Blackwell 区分、直接依赖版本 |
| `tests_mvp/test_backend_launcher.py` | 真实轻量子进程的超时、中断、坏摘要、checkpoint 复检、失败重试和成本去重 |
| `tests_mvp/test_dataset_resume.py` | 同路径替换数据、阶段途中改数据及恢复时的内容锁定 |
| `tests_mvp/test_teacher_export.py` | 导出子进程的超时、失败、重试、恢复与费用 |
| `tests_mvp/test_real_dataset.py` | 实际 RLHFDataset、collate、DataProto 与教师条件的 CPU 接口 |
| `tests_mvp/test_backend_gradients.py` | 真实 PyTorch 梯度：教师信号能改变学生，RL 阶段不依赖教师 |
| `tests_mvp/test_cli.py` | 实际 CLI 的模拟、恢复、计划保存、状态读取和缺少环境变量时报错 |

这些测试验证代码逻辑和接口，不能替代目标机器上的八卡 smoke，也不能证明教学策略提高任务分数。训练结果应读取真实 `run` 输出；`simulate` 输出只用于检查外层流程。
