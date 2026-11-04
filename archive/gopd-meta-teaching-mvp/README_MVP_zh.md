> 历史归档：这是早期G-OPD元教学原型及当时的验证记录。当前G3a入口在 [experiments/student-state-guidance](../../experiments/student-state-guidance/README.md)。下方旧指令不作为当前实验的启动计划。

---

# G-OPD 元教学 MVP

这套代码先验证一个问题：**同一个学生，先接受哪种教师指导，再独立做 RL，最后学得更好？**

训练沿用 G-OPD；新增代码负责选择教法、启动对照分支、比较后续 RL 的结果，以及保存教学经验。上游版本固定为 [`37371a4c`](https://github.com/RUCBM/G-OPD/tree/37371a4c31ad7947746200d234161769191f4748)。

当前已做 CPU 逻辑测试、真实训练命令检查和依赖 metadata 核验。**尚未在 H800、B300 或其他八卡机器上执行完整训练。** GPU 显存、CUDA/NCCL 和收敛效果需要在目标机器上验证。

## 1. 这一版实际做什么

每一轮从同一个学生 checkpoint 出发，运行下面的分支：

```text
同一个学生起点
  ├─ direct_rl：直接做 RL ──────────────────→ 在 meta_val 上评测
  ├─ plain：普通 OPD → 关闭教师 → 独立 RL ──→ 在 meta_val 上评测
  ├─ partial：部分参考 OPD → 关闭教师 → RL ─→ 在 meta_val 上评测
  └─ full：完整参考 OPD → 关闭教师 → RL ────→ 在 meta_val 上评测
                      ↓
           比较“指导分支”和 direct_rl 的分数、成本
                      ↓
           更新小型教学策略，供后续轮次使用
```

第一轮可以枚举全部教法，先收集真实训练结果。后续轮次由策略选择需要尝试的教法。选择用到学生当前表现、剩余预算比例和轮次进度。

**学的是“什么时候选哪种教法”。** 教师模型的权重保持冻结，策略通过改变教师看到的参考内容和被指导样本，改变学生收到的监督。这一版不训练教师语言模型的参数，也不做 token 级长期信用分配。

默认三个动作如下：

| 动作 | 教师看什么 | 哪些学生回答会调用教师 |
|---|---|---|
| `plain` | 原题和学生生成前缀 | 全部回答 |
| `partial` | 原题＋部分参考推理＋学生生成前缀 | 按学生不确定性，选一半回答 |
| `full` | 原题＋完整参考解＋学生生成前缀 | 全部回答 |

`partial` 先去掉 `\boxed{...}` 和显式最终答案尾部，再取剩余参考文本的前 50% 字符。这个规则不保证剩余推理完全不包含答案信息。学生输入始终只有原题。

不确定性来自学生回答的平均负对数概率，**先选样本，再调用教师**。分布式评分需要补齐 batch，实际评分条数可能略高于选中条数，二者都会记录。

默认 `partial` 同时改变参考信息量和查询比例。后续做消融时，应把“只改参考信息”和“只改查询比例”拆成两个动作。

## 2. 拿到八卡机器后怎么开始

把整个工程目录放到机器上，保留 `verl/`、`meta_teaching/`、`mvp_tools/`、`mvp_configs/` 和 `scripts/` 的相对位置。以下命令都在工程根目录执行。

### 已有能跑 G-OPD 的环境：优先使用

先激活现有环境，再检查版本。默认检查不会安装包或启动训练。

```bash
python --version
bash scripts/mvp_environment.sh
```

让该环境使用这一份修改后的 verl：

```bash
python -m pip install --no-deps -e ./verl
```

`--no-deps` 保留现有依赖，不让上游安装配置重新解析整套训练环境。随后检查导入位置、训练依赖、CPU 数据结构和八卡 CUDA 可见性：

```bash
bash scripts/mvp_environment.sh check-training
```

这一步不创建 Ray 集群，不生成回答，也不执行 GPU 训练。若实际导入的是另一份 verl，会停止并指出路径。

### 没有环境：H800/H100 可用这个安装入口

先自己建立一个专用 Python 3.10 或 3.11 的 venv/conda 环境，再执行：

```bash
bash scripts/mvp_environment.sh install-hopper
bash scripts/mvp_environment.sh check-training
```

安装入口只支持 Linux x86_64、计算能力 8.x/9.x 的 NVIDIA GPU，并要求已有 CUDA 12.x `nvcc`。它先运行 pip dry-run，再安装固定的直接依赖，最后编译 FlashAttention。它不安装系统 CUDA，不修改 SSH，不停止其他人的进程。

**B300/B200 不要用 `install-hopper`。** 这两类机器需要已验证支持 Blackwell 的环境。外层教学脚手架仍可使用，但这份旧 torch/vLLM 组合会被安装脚本拒绝。

| 关键组件 | Hopper profile |
|---|---|
| PyTorch / torchvision / torchaudio | `2.6.0` / `0.21.0` / `2.6.0` |
| vLLM | `0.8.5.post1` |
| Transformers / tokenizers | `4.51.3` / `0.21.1` |
| TensorDict / TorchData | `0.8.3` / `0.11.0` |
| Ray | `2.45.0` |
| FlashAttention | `2.7.4.post1`，在安装 torch 后编译 |

原作者安装脚本指定 `tensordict==0.6.2`，但仓库 `setup.py` 要求 `>=0.8.0`，两者冲突。这份 profile 统一用 `0.8.3`；其 TensorDict 基本操作已与 CPU torch 2.6 实测。vLLM 的严格 torch 版本要求来自 [PyPI metadata](https://pypi.org/pypi/vllm/0.8.5.post1/json)，TorchData 对应关系来自 [官方版本表](https://github.com/meta-pytorch/data/blob/v0.11.0/README.md)。

可重新查询依赖约束：

```bash
bash scripts/mvp_environment.sh audit-metadata
```

metadata 一致只表示固定的直接依赖未发现声明冲突；它不证明所有间接依赖、Linux wheel ABI 和 CUDA 内核都可用。这份文件也不是完整的 wheel/hash lock。

## 3. 准备模型与数据

第一组模型建议使用本地 Qwen3-1.7B 学生和 Qwen3-4B 教师。必须验证两者 tokenizer 的 token ID、预处理、特殊 token 和模板一致，不能仅根据“同一家族”推断兼容。

```bash
export STUDENT_MODEL=/data/models/Qwen3-1.7B
export TEACHER_MODEL=/data/models/Qwen3-4B
```

配置通过 `${STUDENT_MODEL}` 和 `${TEACHER_MODEL}` 读取路径。预检只检查本地文件，不自动下载模型。

先用演示数据验证流程：

```bash
python -m meta_teaching.data --demo --output data/mvp
```

这会生成三份互不重叠的简单加法数据和 `manifest.json`。**它们仅用于 smoke test，不能支持研究结论。**

真实实验准备三个独立 JSONL 文件：

| 文件 | 用途 | 必需字段 |
|---|---|---|
| `train.jsonl` | 学生 OPD/RL 训练 | `question`、`answer`、`reference_solution` |
| `meta_val.jsonl` | 衡量后续 RL 的收益，更新教学策略 | `question`、`answer` |
| `test.jsonl` | 最后评估选定模型 | `question`、`answer` |

训练样本示例：

```json
{"id":"train-001","question":"Calculate 12 + 17. Put the final answer in \\boxed{}.","answer":"29","reference_solution":"12 + 17 = 29, so the answer is \\boxed{29}."}
```

转换命令：

```bash
python -m meta_teaching.data \
  --train /data/math/train.jsonl \
  --meta-val /data/math/meta_val.jsonl \
  --test /data/math/test.jsonl \
  --output data/research
```

转换器检查样本 ID 和归一化问题是否重复，保存文件哈希与 split 清单。训练参考解只进入 teacher metadata；验证/测试中的参考解字段会被丢弃。`answer` 用于奖励计算，不放入学生问题。

正式数据在配置中设 `data.manifest` 为 `data/research/manifest.json`，设 `data.reward` 为 `math_verify`。`numeric_demo` 只用于演示数字题；缺少正式验证器时不会自动降级。

## 4. 先检查计划，再运行

四份配置按这个顺序使用：

| 配置 | 用途 | 已设置的规模 |
|---|---|---|
| `mvp_configs/simulation.json` | 不加载模型，检查外层流程 | 3 轮、每动作 2 个 repeat；模拟分数 |
| `mvp_configs/smoke_8gpu.json` | 八卡上确认能采样、更新、切阶段、保存 | 1 轮、1 个 repeat；OPD 1 步＋RL 2 步；演示数据 |
| `mvp_configs/pilot_8gpu.json` | 换成真实数学题，观察不同教法的后续效果 | 3 轮、3 个 repeat；OPD 5 步＋RL 20 步；固定步数 |
| `mvp_configs/budget_8gpu.json` | 在时间配额下比较方法 | 3 轮、3 个 repeat；每分支 3600 秒；固定时间 |

`pilot` 和 `budget` 默认读取 `data/research/manifest.json`。三份真实训练配置都需要前面的模型环境变量；模拟配置不需要。

```bash
python -m meta_teaching plan \
  --config mvp_configs/smoke_8gpu.json \
  --save runs/smoke-plan.json
python -m meta_teaching preflight --config mvp_configs/smoke_8gpu.json --gpu
```

`plan` 把分支、训练阶段和命令保存为 JSON；不写 `--save` 时放在配置输出目录的 `plan.json`。`preflight --gpu` 检查模型文件、tokenizer、三份数据及哈希、训练依赖和 GPU 环境。省略 `--gpu` 可以先做本地检查；`run` 会再次执行严格 GPU 预检。预检成功也不等于训练已成功。

确认预检通过后，运行第一组 GPU smoke：

```bash
python -m meta_teaching run \
  --config mvp_configs/smoke_8gpu.json \
  --output-dir runs/smoke-001
```

查看状态：

```bash
python -m meta_teaching status --output-dir runs/smoke-001
```

只想先检查外层调度、不占用 GPU，可以使用：

```bash
python -m meta_teaching simulate --output-dir runs/simulated-001
```

`simulate` 默认使用 `simulation.json`，**无需设置模型环境变量，也无需下载模型或先准备演示数据**。它使用明确标记的模拟后端，检查计划、记账、选策和恢复。生成的分数不是模型性能，不能与真实训练结果合并。

已完成阶段的结果可以复用；继续同一实验使用原配置和目录：

```bash
python -m meta_teaching run \
  --config mvp_configs/smoke_8gpu.json \
  --output-dir runs/smoke-001 \
  --resume
```

**单独的 `--resume` 不会自动重跑失败阶段。** 先用 `status` 看实验记录，再看该阶段的 `train.log`、`launcher_record.json` 和 checkpoint；真正继续时程序也会检查旧进程是否还活着。确认需要重试后显式执行：

```bash
python -m meta_teaching run \
  --config mvp_configs/smoke_8gpu.json \
  --output-dir runs/smoke-001 \
  --resume --retry-failed
```

这会把失败目录归档为 `*.failed-时间戳`，旧尝试仍计入成本，然后**从该阶段原来的输入 checkpoint 重做整个阶段**。它不从失败阶段中途写出的 partial checkpoint 续训。已成功完成的阶段继续复用，不重复执行。

保留旧日志、checkpoint 和账本；不要删除记录后把失败开销当成从未发生。无法确定硬中断结束时间时，恢复过程按保守上界计费，并标注估算成本。

## 5. 怎么判断这一版是否正常

先检查流程，再讨论效果：

1. 每个指导分支先结束 warmup 进程，再开启独立 RL 进程。
2. RL/eval 的 `teacher_enabled` 为 `false`，`teacher_queries` 为 `0`。
3. partial/full 的参考信息只出现在教师输入中，学生回答 token 不重新编码。
4. 后续 RL 从完整 checkpoint 恢复，保留模型、optimizer、scheduler、训练 RNG 和 rollout RNG 文件。
5. 教学策略的更新标签来自 **RL 结束后的 meta_val 分数**，不能用 warmup 分数代替。
6. 所有实际执行的分支都进入成本账本；最终只对选定模型做一次 test 评测。

checkpoint 保留随机状态，但阶段切换会重建 vLLM 推理引擎，**不承诺逐位重放完全相同的采样结果**。正式实验需要多个训练种子。

主要结果文件：

| 文件 | 直接看什么 |
|---|---|
| 每阶段 `phase_request.json` | 实际运行参数和 Python 命令 |
| 每阶段 `train.log` | 训练日志、OOM、依赖异常 |
| 每阶段 `phase_summary.json` | 完成步数、最终验证分数、教师调用、checkpoint |
| 每阶段 `launcher_record.json` | 进程状态、开始/结束时间、实际占用 GPU 秒数、是否超时 |
| `experiment_state.json` | 已完成分支、当前轮次、教学策略及继续运行状态 |
| `results.json` | 全部实验结果、成本和最后的 test 结果 |
| `branches.csv` | 每个分支的 warmup/post-RL 分数、步数、成本和排除原因 |
| `teaching_policy.json` | 可继续使用的动作价值模型与策略随机状态 |

## 6. 两种预算模式分别说明什么

### `fixed_steps`：先排查训练逻辑

直接 RL 分支运行 `W + H` 步；指导分支先 OPD `W` 步，再 RL `H` 步。

**这不是等算力比较。** OPD 的教师评分有额外成本；warmup 与 RL 的 rollout 数也不同。这种模式用于判断流程、损失和学习信号是否正常，不能直接写成“固定总预算优于直接 RL”。

### `fixed_walltime`：比较相同的时间配额

每个分支拿到相同的开始到截止时间，指导分支在这一配额里分配 warmup 和 RL。每个 batch 边界检查是否要停止，并预留评测时间。

一个 batch、最终评测或保存可能越过截止时间，实际超时会记录。超出设定容忍比例的配对不用于策略更新。全局预算不足以提供一个分支的完整剩余配额时，会停止，不把缩水配额当成同预算实验。预算配额不是零误差的 GPU 秒数配平，分析时仍应读取实际成本。

GPU 成本采用 **分配卡数 × 整个子进程运行时间**，覆盖加载、teacher、采样、训练、评测、保存和退出。额外候选分支也算成本，外层控制器 CPU 时间单独记录。这是资源占用统计，不是逐算子的 FLOPs。

教学动作的训练标签为：

```text
收益 = 指导后独立 RL 的 meta_val 分数 − 直接 RL 的 meta_val 分数
       − 每 GPU 小时的成本惩罚 × 两个分支的 GPU 小时差
```

`cost_penalty_per_gpu_hour=0` 时只学习分数差。分数本身越高越好，成本系数需要使用相同的分数尺度。对所有候选进行搜索产生的总开销仍计入全局账本，不能只报告被选中分支的成本。

## 7. 后续怎么迭代我们的 idea

先改 `mvp_configs/` 中的动作，不急着改训练底层。比如保持所有动作 `query_fraction=1`，只比较教师看到 0%、25%、50%、100% 的参考内容；这样可以先分清信息量的作用。

当前策略是每个动作一个小型线性回归，并使用 epsilon-greedy 探索。它根据真实配对结果估计后续收益，不能把自己的预测当作训练标签。要换更强的价值模型，入口是 `meta_teaching/policy.py`。

默认 `teacher_mode="fixed"`：教师能力底座不变，保留教学策略和估计经验。`teacher_mode="previous_student"` 可把上一轮选定学生的 HF 导出作为下一轮教师；它用于检查自举流程，不能证明递归改进已经有效。

默认 `promotion="policy"` 在本轮结果出来前决定要继续使用的动作，避免看完结果后挑最好分支却把它算作策略能力。多个 repeat 中固定使用第一个种子的对应 checkpoint 进入下一轮，不能事后选择最好的随机种子。`validation_best` 属于显式搜索基线，应单独报告其全部搜索成本。

下一步最值得加的实验是：固定教法、只学教法、只学查询率、两者都学；以及保留/重置教学策略的跨轮对照。多步 agent、可训练教师权重、token 级 credit 和复杂 RSI 不在这一版范围内。

## 8. 本地测试

轻量 CPU 环境只需要：

```bash
python3.11 -m venv .venv-mvp-cpu
source .venv-mvp-cpu/bin/activate
bash scripts/mvp_environment.sh install-cpu
python -m pytest tests_mvp -q
```

缺少 torch、numpy、tensordict、Ray 等训练侧依赖时，对应集成测试会明确显示 `skipped`；其余测试仍执行。要覆盖实际 TensorDict 操作和 trainer hooks，请在已有 G-OPD 环境或完整训练环境中运行同一命令：

```bash
python -m pytest tests_mvp -q
```

文件级修改位置见 [MVP_FILE_MAP.md](MVP_FILE_MAP.md)。

## 数据、恢复与最终模型
恢复时会检查 manifest 和三份 Parquet 的内容哈希。不要在同一个 run 目录下换数据、改 split 或改配置；新实验使用新输出目录。阶段运行途中数据发生变化，该阶段也会被判为失败，已消耗的费用仍保留。
如果控制器被强制杀死，无法确定遗留进程的准确结束时间，恢复会以恢复时刻计算该次尝试的保守费用上界，并在结果中标记 estimated_upper_bound_cost_events。它不是精确实测 GPU 时间。
默认保留完整 FSDP checkpoint，便于下一轮训练。要单独导出最终学生供推理使用，可读取 results.json 的 selected_checkpoint，再运行：
```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir /实际/selected_checkpoint/actor \
  --target_dir /data/models/meta-teaching-student
```
运行前让 Python 使用本工程的 verl（按上文 editable 安装，或进入工程的 verl/ 目录）。请将示例路径替换为实际路径。previous_student 模式会自动执行同类导出，并为导出子进程单独记账、保留失败记录。
训练 RNG 和 dataloader 在分支起点保持相同；repeat 会改变推理引擎初始化种子。它们是配对的采样重复，不保证每一种训练随机流都完全独立。
