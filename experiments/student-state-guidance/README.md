# K&K：四种指导×两个学生起点

当前只做K&K、seed42。从第30、90步学生存档，各比较Baseline、只给答案、完整解法、针对错误提示，**八条正式分支各追加150步**。共用准备路线提供起点，不算八组之一。

[实验在测什么](../../docs/g3a-study.md) · [进度与修复结果](../../docs/g3a-results.md)

## 代码与修复

运行核心对应`0ee2ff2bebe692fdc5d428db5ae7cb25551ae2c4`，基于[StringNLPLAB/opd-rlvr](https://github.com/StringNLPLAB/opd-rlvr)。当前修复：

- 对齐老师与学生的结束符概率，真实结束位置保留OPD监督，padding仍屏蔽。
- 用Hydra的字符串编码器传命令行参数，保留真实换行。训练和终测共享提示格式。
- 30/90起点保存完整状态；分支中途只写日志，150步终点保存模型，不保存优化器。

额外的老师格式提醒仅保留为显式诊断选项，当前正式配置关闭。Baseline没有额外答案或解法，但仍使用老师对原题和学生前缀的token概率。

| 目录 | 内容 |
|---|---|
| `state_guidance/` | 数据、teacher-only指导、调度、终测、统计与修复测试 |
| `configs/kk8-scope.json` | 当前八组范围＋共用参考路线 |
| `scripts/run_g3.sh` | 训练与终测守候入口 |
| `tools/` | 只读已有输出并导出表格、曲线 |
| `logic/`、`verl/` | 保留许可证的上游实现与当前补丁 |
| `reproduction/`、`guidance/` | 复用配置和证明函数；历史协调器不是当前入口 |

`core18-scope.json`和原全矩阵协议是历史来源，不是默认队列。

## 准备环境与输入

使用兼容的Linux环境。参考版本是Python3.10、torch2.7.1、vLLM0.10.1.1、Transformers4.55.4；[环境记录](configs/runtime-reference.json)不是跨机器通用的驱动锁。

```bash
python -m pip install --no-deps -e ./verl
python -m pip install --no-deps -e ./logic
export PYTHONPATH="$PWD/verl:$PWD/logic:$PWD/logic/training:$PWD"
export REPRO_ROOT=/absolute/path/to/new-persistent-run
export G3_MODEL_ROOT=/absolute/path/to/model-cache
python -m state_guidance.public_setup init --root "$REPRO_ROOT"
python -m state_guidance.public_setup plan --root "$REPRO_ROOT"
```

`init`只写新目录，发现旧运行状态或配置冲突就拒绝覆盖。它写入当前范围、结束符修复、存储策略和提示格式；不会启动训练。

模型不在Git中。按[固定revision](configs/model-revisions.json)下载到`MODEL_SOURCE`后，核验23个官方文件SHA并建立缓存：

```bash
export MODEL_SOURCE=/absolute/path/to/downloaded-models
python -m state_guidance.bootstrap_cache --manifest configs/model-files.json   --source "$MODEL_SOURCE" --cache "$G3_MODEL_ROOT"   --state "$REPRO_ROOT/preflight/model-cache.json"
python -m state_guidance.public_setup fetch-data --root "$REPRO_ROOT"
python -m state_guidance.prepare_data --root "$REPRO_ROOT" --task kk   --raw-data "$REPRO_ROOT/source_data" --workers 8
python -m state_guidance.prepare_guidance --root "$REPRO_ROOT" --task kk --workers 8
python -m state_guidance.audit_prompts --root "$REPRO_ROOT" --task kk   --tokenizer "$G3_MODEL_ROOT/Qwen3-8B"
python -m state_guidance.public_setup check --root "$REPRO_ROOT" --models "$G3_MODEL_ROOT"
```

来源包保留冻结来源清单，实际只生成和训练K&K。数据、指导和提示审计的预期SHA在[frozen-inputs.json](configs/frozen-inputs.json)。已存在的不同文件不会被静默替换；`.partial`不是完整资产。

## 启动、存档与状态

两条独占4GPU路线就绪后：

```bash
bash scripts/run_g3.sh train
# 另一终端：先等待完整训练结束，再做固定终点评测。
bash scripts/run_g3.sh finalize
```

共8张GPU，两组并行；30步起点四组完成后才进入90步起点。默认batch128、rollout8、mini16、学习率1e-6、回答上限4096。每25步开发集512×8，50/100/150报告；终点测试1024×32另表。

公开从头运行会用修复版训练出新的准备存档。当前部署复用了历史普通OPD的完整30/90存档；精确重现其数值需要那些未公开的模型、优化器、RNG和sampler状态。两种起点来源须分别标注。

已有成功路线可复用；失败和迁移保留日志与成本。迁移续训用`migration`准备后显式`--resume-migration`，普通失败重试从原起点重跑。只含终点模型的存档不能恢复优化器训练。更新发布副本不等于更新活跃release；不要覆盖正在运行的代码。

进度以真实进程、`launcher_record.json`、连续训练metric及完整开发输出为准。step0是起点成绩。导出已有结果：

```bash
python tools/export_progress.py --root "$REPRO_ROOT" --out "$REPRO_ROOT/reports"
```

CPU回归入口是`bash scripts/test_g3_cpu.sh`，不加载训练模型、不做GPU生成。来源、公开路径修改与验证记录见[发布说明](PUBLICATION_PROVENANCE.json)。原README和许可保留为[UPSTREAM_README.md](UPSTREAM_README.md)及LICENSE/NOTICE。
