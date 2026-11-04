> 历史归档：这是早期G-OPD元教学原型及当时的验证记录。当前G3a入口在 [experiments/student-state-guidance](../../experiments/student-state-guidance/README.md)。下方旧指令不作为当前实验的启动计划。

---

# B300 执行交接：G-OPD 元教学 MVP
交付日期：2026-10-18。方法版本：MVP v0.1.0。本次发布到 GitHub，并将阅读笔记与实验源码分开；训练代码保持原交付版。
**当前状态：代码可交接；B300 训练环境和真实 8 卡训练待验证。** 本包没有模型权重、正式训练数据或已验证的 B300 依赖锁。已有 CPU 测试记录为 137 passed；这不代表 CUDA、NCCL、FSDP、vLLM 已验证。

## 1. 接手目标
先在 B300 上完成一次真实 8 卡 smoke，并交回环境、日志和结果。先复用已验证支持 B300 的环境，在独立环境或容器中适配本仓库。完整 smoke 成功后固定依赖版本；长时间 pilot 是后续实验，当前交接先完成 smoke。
上游是 [RUCBM/G-OPD](https://github.com/RUCBM/G-OPD/tree/37371a4c31ad7947746200d234161769191f4748)，固定 commit `37371a4c31ad7947746200d234161769191f4748`，内含 verl v0.6.1。只 clone 上游不会得到本项目新增的教学选择器和实验调度。
本目录是完整源码快照，作为主仓库的普通子目录管理，不需要初始化子模块。来源见 `MVP_UPSTREAM.json`；上游各项许可证保留在原位置。

## 2. 从 GitHub 获取仓库
在 B300 上执行：
```bash
git clone https://github.com/Tasselszcx/opd-rl-research.git
cd opd-rl-research/archive/gopd-meta-teaching-mvp
git status --short
git log -1 --oneline
```
后续所有训练命令都在这个 MVP 子目录运行。若主仓库已 clone，直接进入该目录；保留已有修改，不重复覆盖 checkout。
阅读笔记位于主仓库的 `reading-notes/`，训练代码位于 `archive/gopd-meta-teaching-mvp/`。`START_B300_CODEX.txt` 是可直接发给远端 Codex 的接手指令。GitHub 版本不需要先传离线 bundle，也不需要再 clone G-OPD 上游。
仓库不包含 Python 虚拟环境、模型权重、正式训练数据或训练 checkpoint。

## 3. B300 环境选择
**不要执行 `bash scripts/mvp_environment.sh install-hopper`，也不要照抄 `mvp_requirements_hopper.txt`。** 它们固定 torch 2.6 / vLLM 0.8.5.post1，安装脚本会主动拒绝 Blackwell。
B300 的计算能力是 10.3，B200 是 10.0。CUDA 12.9 新增 `sm_103` 编译目标；若编译原生 sm_103 内核，应使用 CUDA 12.9 或更高的适配工具链。预编译包是否可运行仍要看其二进制兼容性及实测，不能只根据一个 CUDA 版本号判定。[NVIDIA GPU 表](https://developer.nvidia.com/cuda/gpus)、[CUDA 12.9 发布说明](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html#general-cuda)。
这份旧 verl 使用 vLLM 的 `external_launcher`、sleep/wake、权重加载和内部 worker/model_runner 接口。较新的 vLLM 能在 B300 推理，不自动证明它能与本项目一起训练。先检查现有环境，遇到接口问题再做最小适配并保留 diff；不要无依据地全量升级或降级。
需要重点检查的文件：
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/verl/workers/sharding_manager/fsdp_vllm.py`
- `verl/verl/utils/vllm/utils.py`
- `verl/verl/workers/fsdp_workers.py`
在目标机器记录 GPU 名称、驱动、可用卡、Python 路径、torch/CUDA/vLLM/FlashAttention/Transformers/Ray/TensorDict/TorchData 版本。检查当前作业占用，使用分配给本实验的 8 张卡；不要终止其他任务。环境清单不要包含令牌、密码或私有源认证信息。
```bash
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,utilization.gpu,driver_version --format=csv
python --version
python -c 'import sys; print(sys.executable)'
```
激活选定的项目专用环境后，统一环境检查与训练使用的 Python：
```bash
export MVP_PYTHON="$(command -v python)"
bash scripts/mvp_environment.sh check
python -m pip install --no-deps -e ./verl
bash scripts/mvp_environment.sh check-training
```
`--no-deps` 只让当前环境导入本项目的 verl，不自动安装其余依赖。若缺依赖，在项目专用环境中按兼容约束补齐；不要用 Hopper 文件覆盖已有 B300 torch/vLLM 栈。预检对 torch 2.6/vLLM 0.8.5 的提示来自上游旧配置，在 B300 上不要据此降级。
`check-training` 检查导入、CPU 数据结构和 CUDA 可见性；不执行 GPU 生成、反向传播或 NCCL。成功只意味着可以继续 smoke。

## 4. 模型与首次配置
用已下载的本地模型，先采用小规模 student/teacher。下面的 Qwen3 路径是占位示例，需替换为实际目录：
```bash
export STUDENT_MODEL=/data/models/Qwen3-1.7B
export TEACHER_MODEL=/data/models/Qwen3-4B
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```
每个目录需有权重、`config.json`、`tokenizer.json` 及 chat template。两者的词表 ID、tokenizer 处理与模板须通过项目严格检查，同系列名称不足以保证兼容。不要用完全相同权重的模型替代这一组来验收普通 OPD 信号。
默认 `mvp_configs/smoke_8gpu.json` 已是单机 8 卡，不需要额外建一个“B300 配置”。`rollout_tp=2` 只表示生成阶段的张量并行度，训练仍占用 8 卡。四个分支依次运行。
| 配置 | smoke 初始值 | 意义 |
|---|---:|---|
| `training.n_gpus` | 8 | 单机可见卡数 |
| `training.rollout_tp` | 2 | vLLM 生成并行度 |
| `training.train_batch_size` | 16 | 每批题数 |
| `training.ppo_mini_batch_size` | 16 | 更新 mini batch |
| `training.micro_batch_size_per_gpu` | 1 | 每卡 micro batch |
| `training.warmup_steps` / `rl_steps` | 1 / 2 | 指导分支的新训练 batch 数 |
| `experiment.rounds` / `repeats` | 1 / 1 | 首轮完整流程验收 |
| `experiment.max_total_gpu_hours` | 8 | 全实验累计资源上限，约一小时八卡占用，不是耗时承诺 |
若加载较慢使预算不足，保留真实开销；调整预算后使用新 run 目录，不将未完成实验标记为成功。改数据、配置、模型或源码后重新验收时，应使用新目录并记录变更。

## 5. 跑一轮真实 smoke
以下命令均在工程根目录、选定环境内执行：
```bash
python -m meta_teaching.data --demo --output data/mvp
python -m meta_teaching plan \
  --config mvp_configs/smoke_8gpu.json \
  --output-dir runs/b300-smoke-001 \
  --save runs/b300-smoke-plan.json
python -m meta_teaching preflight \
  --config mvp_configs/smoke_8gpu.json \
  --output-dir runs/b300-smoke-001 --gpu
bash scripts/run_mvp_8gpu.sh \
  mvp_configs/smoke_8gpu.json \
  --output-dir runs/b300-smoke-001
python -m meta_teaching status --output-dir runs/b300-smoke-001
```
使用有权限的终端会话管理工具保留长进程；不要同时对同一个 run 启动多个控制器。演示数据是简单加法，仅用于流程验收。
此配置应完成：
```text
同一 student 起点
  direct_rl : RL 3 步
  plain     : OPD 1 步 → 关闭 teacher → RL 2 步
  partial   : OPD 1 步 → 关闭 teacher → RL 2 步
  full      : OPD 1 步 → 关闭 teacher → RL 2 步
比较各分支最终 meta_val → 更新选择器 → 对预先选定分支做最终 test
```
总计 4 个分支、7 个训练子进程、初始和最终各 1 个纯评测子进程。一次 smoke 只能验收策略更新，无法证明学到的策略对下一轮有帮助。

## 6. 验收依据
1. `runs/b300-smoke-001/results.json`：`status=completed`、`simulated=false`、`rounds_completed=1`、`final_test` 非空；报告实际预算和超限值，正常完整验收应无超限。
2. `branches.csv` 有 4 行有效分支。direct_rl 的 `rl_steps=3`；另外三行 `warmup_steps=1`、`rl_steps=2`；`valid_for_policy_update=true`。
3. 各 warmup 的 `phase_summary.json`：`teacher_enabled=true`，`teacher_queries>0`、`teacher_response_tokens>0`。分布式 padding 可能使实际查询数略高于选中样本数。
4. 各 RL/eval 的 summary：`teacher_enabled=false`，`teacher_queries`、`teacher_selected_queries`、`teacher_scored_tokens`、`teacher_response_tokens` 都为 0。
5. `plain/partial/full` 的 RL 从对应 warmup 的完整 checkpoint 恢复；`direct_rl` 从本轮共同起点开始，首轮为原始 student。核对 `source_checkpoint`、`start_step`、`end_step`、`steps_completed`、`validation_step`。保留 8 rank 的模型、优化器、额外状态和 rollout RNG 文件，以及 dataloader 状态 `data.pt`。
6. `train.log` 中 loss/grad 数值有限；不同模型的 OPD 应出现有效梯度。仅有 `steps_completed` 不足以证明参数改变。加法 demo 中 RL 奖励可能全相同，零 RL 梯度不直接说明代码坏了；如需确认 RL 学习，另用存在成败混合的题目检查梯度及 checkpoint 张量变化。
7. `results.json` 的 `policy_diagnostics.total_updates=4`，`experiment_state.json` 的 `rounds[0].feedback` 有三条教学反馈；标签来自后续 RL 分数及成本，不能用 warmup 分数替代。
8. final test 只做一次，使用选定 student；测试分数不参与教法选择器更新。
**不要把 simulate、预检通过、进程退出码 0 或加法得分单独写成“B300 已跑通”。** 验收对象是生成、teacher 评分、OPD 更新、保存、独立 RL、评测和选择器更新的完整链路。

## 7. 恢复与失败定位
成功运行后用同一配置、模型、数据和目录做一次恢复检查：
```bash
bash scripts/run_mvp_8gpu.sh \
  mvp_configs/smoke_8gpu.json \
  --output-dir runs/b300-smoke-001 --resume
```
应该复用已完成阶段，不再启动训练或重复计入阶段费用。外层控制器自身开销可能略变，不要求全部 JSON 逐字相同。
失败先读取该阶段的 `train.log`、`launcher_record.json`、`phase_request.json`。修复原因后，如实验输入未变、确实需要重试该阶段：
```bash
bash scripts/run_mvp_8gpu.sh \
  mvp_configs/smoke_8gpu.json \
  --output-dir runs/b300-smoke-001 --resume --retry-failed
```
该命令保留失败记录和开销，从失败阶段原输入 checkpoint 重跑整个阶段；不是从 partial checkpoint 续训。依赖或源码兼容性修改后的重新验收，优先新 run，避免混合不同实现的结果。
| 故障 | 首先检查 |
|---|---|
| vLLM import / worker 属性错误 | 上述三个 vLLM 适配文件与实际安装版本 |
| no kernel image / invalid device function | torch、CUDA 扩展及 FlashAttention 对 B300 的支持 |
| tokenizer 检查失败 | 本地词表、模板、特殊 token，不要直接绕过检查 |
| 显存不足 | 实际显存占用、模型规模、长度及 batch，保留所有对照同等配置 |
| RL 仍调用 teacher | phase 请求与 summary，保持 teacher-free 契约 |
| 预算耗尽 | 所有分支、加载、评测、失败尝试都计费，查看成本账本 |

## 8. 方法边界与后续实验
当前 teacher 权重冻结；小型选择器学习在学生分数、剩余预算、轮次条件下选 `direct_rl/plain/partial/full`。partial 同时改变参考信息量和查询率，正式消融要拆开。学生只看到题目，参考解只用于 teacher 输入。默认 teacher 固定；`previous_student` 可检查跨轮 teacher 更换，但尚无收益证据。
保持这些边界：teacher 先根据学生信号选查询样本；OPD 使用 student 自己生成的 token；后续 RL 无 teacher；配对分支从同一 checkpoint 出发；本轮接续分支在看到本轮结果前选定；所有搜索和失败成本进入账本。
正式数学实验需要独立的 train/meta_val/test。训练集字段为 `question, answer, reference_solution`，其他两份为 `question, answer`：
```bash
python -m meta_teaching.data \
  --train /data/math/train.jsonl \
  --meta-val /data/math/meta_val.jsonl \
  --test /data/math/test.jsonl \
  --output data/research
```
后续启动命令：
```bash
bash scripts/run_mvp_8gpu.sh mvp_configs/pilot_8gpu.json --output-dir runs/b300-pilot-001
```
pilot 是 3 轮、3 次重复、OPD 5＋RL 20，对照为 RL 25，总预算 72 GPU 小时（约 9 小时八卡占用上限）。这只对齐训练步数，不等于等算力。固定时间比较用 `budget_8gpu.json`，每分支 3600 秒，所有尝试合计上限 220 GPU 小时。不要将这些长实验混入本次 smoke 的验收。

## 9. 执行端交回哪些文件
- `B300_RUN_REPORT.md`：实际主机/GPU、源码 commit、模型、环境、各阶段结果、成本、已知问题和下一步。
- 可复现的环境版本清单；若使用容器，记录镜像 tag/digest。分享前移除认证信息和私人 URL。
- 新增适配的 Git diff/commit；通过 smoke 后固定环境，说明每项变更原因。
- `preflight.json`、`results.json`、`branches.csv`、`teaching_policy.json`、`experiment_state.json`。
- 所有阶段的 `phase_request.json`、`phase_summary.json`、`launcher_record.json`、`train.log`；失败阶段也保留。
- 首次运行和成功后 resume 的对比结论；完整 checkpoint 留在服务器，报告可访问路径。
补充资料：`README_MVP_zh.md`、`MVP_FILE_MAP.md`、`VALIDATION_REPORT.md`、`validation/report.json`。这些旧验证记录都保留其原始范围，不能改写成新的 GPU 测试结果。
