# G3a：固定老师的学生状态与指导实验

这份代码比较：训练30步和90步的学生，分别接受四种teacher提示，再继续训练150步。当前只执行K&K、GSM8K、训练种子42，共18条路线。

[实验设计](../../docs/g3a-study.md) · [已有开发结果](../../docs/g3a-results.md) · [发布验证](../../docs/validation.md)

## 先看目录

| 目录 | 作用 |
|---|---|
| `state_guidance/` | 当前G3a的数据、teacher-only提示、训练调度、迁移、终测与统计 |
| `configs/` | 当前18条范围、原协议、冻结输入SHA、模型文件与运行环境版本 |
| `scripts/run_g3.sh` | 完整输入核验后，显式启动训练或终测守候 |
| `scripts/test_g3_cpu.sh` | CPU协议与恢复测试，不加载模型、不做GPU生成 |
| `tools/` | 只读已有开发输出，导出表格与曲线 |
| `reproduction/` | 早期复现代码与当前仍复用的配置函数 |
| `guidance/`、`length_audit/` | 历史指导实验工具；部分证明函数仍被当前数据构建复用 |
| `verl/`、`logic/`、`math/` | 带当前修改的上游实现，许可证保留 |

本研究使用`state_guidance`入口。其他目录的旧协调器不是当前18条实验的启动方式。

## 环境

训练面向Linux和兼容的CUDA环境。参考版本为Python 3.10.16、torch 2.7.1、vLLM 0.10.1.1、Transformers 4.55.4、FlashAttention 2.8.3、Ray 2.49.1、TensorDict 0.9.1和NumPy 1.26.4。完整核心版本记录见 [runtime-reference.json](configs/runtime-reference.json)。它是参考部署的版本记录，不是跨机器通用的驱动或容器锁。

先激活兼容的独立环境，再让Python导入本目录的代码：

```bash
python -m pip install --no-deps -e ./verl
python -m pip install --no-deps -e ./logic
export PYTHONPATH="$PWD/verl:$PWD/logic:$PWD/logic/training:$PWD"
```

环境包装器使用当前激活的Python。只有复用迁移环境归档时才设置`MIGRATION_ROOT`；默认使用系统原生驱动，`G3_USE_CUDA_COMPAT=1`是显式兼容模式，不自动注入compat库。

## CPU检查与18条计划

CPU测试需要Linux；进程身份测试依赖`/proc`。可在独立CPU环境中执行：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
bash scripts/test_g3_cpu.sh
```

初始化一个**新的**持久输出目录，再查看18条计划。这两个命令不启动训练：

```bash
export REPRO_ROOT=/absolute/path/to/persistent/g3a-run
python -m state_guidance.public_setup init --root "$REPRO_ROOT"
python -m state_guidance.public_setup plan --root "$REPRO_ROOT"
```

`init`遇到不同的旧配置或已有执行状态会拒绝覆盖。已有实验继续使用其原始`study_plan.json`、`ACTIVE_SCOPE.json`、checkpoint和账本，不重新初始化。

## 准备冻结输入

Qwen模型权重和运行数据不在Git里。模型仓库与固定revision见 [model-revisions.json](configs/model-revisions.json)，23个文件的官方SHA见 [model-files.json](configs/model-files.json)。先把对应revision下载到`MODEL_SOURCE`，再验证和复制到本地缓存：

```bash
export MODEL_SOURCE=/absolute/path/to/downloaded-models
export G3_MODEL_ROOT=/absolute/path/to/local-model-cache
python -m state_guidance.bootstrap_cache \
  --manifest configs/model-files.json --source "$MODEL_SOURCE" \
  --cache "$G3_MODEL_ROOT" --state "$REPRO_ROOT/preflight/model-cache.json"
```

`MODEL_SOURCE`下分别是`Qwen3-8B/`和`Qwen3-1.7B-Base/`。校验失败不会替换已有不同文件；`.partial`不代表完整资产。

以下步骤下载固定版本的公开源数据，构建相同划分、指导和提示审计，均不请求GPU生成：

```bash
python -m state_guidance.public_setup fetch-data --root "$REPRO_ROOT"
for task in kk gsm8k; do
  python -m state_guidance.prepare_data --root "$REPRO_ROOT" --task "$task" \
    --raw-data "$REPRO_ROOT/source_data" --workers 8
  python -m state_guidance.prepare_guidance --root "$REPRO_ROOT" --task "$task" --workers 8
  python -m state_guidance.audit_prompts --root "$REPRO_ROOT" --task "$task" \
    --tokenizer "$G3_MODEL_ROOT/Qwen3-8B"
done
python -m state_guidance.public_setup check --root "$REPRO_ROOT" --models "$G3_MODEL_ROOT"
```

数据源固定commit及SHA见 [public-data-sources.json](configs/public-data-sources.json)。冻结划分和指导SHA、全量提示审计见 [frozen-inputs.json](configs/frozen-inputs.json)。`check`核验数据、范围、模型验证记录及文件大小；完整模型SHA由前面的`bootstrap_cache`核验，另加`--hash-models`可重新计算。

已有冻结输入应复用并核SHA。`prepare_guidance`会保留已有指导文件并拒绝覆盖，不把它当作可随时重生成的临时文件。

## 启动与接续

输入完整后，在两条独占4卡路线均可使用时分别启动训练与终测守候：

```bash
bash scripts/run_g3.sh train
# 在另一终端运行；它先等完整训练矩阵，不会与训练抢同一条路线。
bash scripts/run_g3.sh finalize
```

训练会先执行一次必要更新和续训入口检查，再跑固定150步队列。已有成功路线会复用。不能因开发分数低而重试、缩短条件或改选checkpoint。

状态依据包括`study_state.json`、各`launcher_record.json`、真实进程身份、增长的`train.log`和`checkpoints/metrics.jsonl`。仅有进程或空闲GPU不等于完成了新更新。

失败、用户中断、取消和迁移要分别保留日志与成本。`--retry-failed`从失败路线原起点重跑；迁移断点续训使用`state_guidance.migration`准备并显式`--resume-migration`，恢复模型、优化器、RNG和`data.pt`，不重新重置sampler。`--adopt-running`仅用于核验身份后的调度器接管，不是普通启动选项。

不要把新的代码整理副本覆盖到活跃训练release。换机或更新运行版本时，先核验资产、身份、检查点和已结算成本，再按恢复流程操作。

## 导出已有进度

```bash
python tools/export_progress.py --root "$REPRO_ROOT" --out "$REPRO_ROOT/reports"
```

输出逐路线进度、全部开发点、50/100/150节点CSV与PNG/SVG曲线。只读已有训练/开发输出，不加载模型、不重采样。训练reward与开发正确率分开；未完成验证的点保留为待验证。完整终测和配对统计由`state_guidance.finalize`、`state_guidance.analyze`完成。

## 来源与发布边界

训练主体来自固定快照`140f0cb5935637c7ccc500240da9c786056424cc`，其上游为 [StringNLPLAB/opd-rlvr](https://github.com/StringNLPLAB/opd-rlvr)。原README保留为 [UPSTREAM_README.md](UPSTREAM_README.md)，许可证与NOTICE保留。

本次增加可移植路径、公开输入准备与核验入口，未改变当前18条生成的科学训练配置；差异与来源记录在 [PUBLICATION_PROVENANCE.json](PUBLICATION_PROVENANCE.json)。原部署已有部分真实GPU训练记录；本整理版做CPU回归，不把它说成全部18条和终测已完成。
