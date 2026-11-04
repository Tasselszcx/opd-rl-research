# 实验代码

| 项目 | 用途 | 接手入口 |
|---|---|---|
| [gopd-meta-teaching-mvp](gopd-meta-teaching-mvp/) | OPD warmup → 独立 RL → 用后续收益更新教法选择器 | [B300 handoff](gopd-meta-teaching-mvp/HANDOFF_B300.md) · [Codex 指令](gopd-meta-teaching-mvp/START_B300_CODEX.txt) |

这里保存可编辑源码，采用主仓库内的普通目录。当前完整交付一个 MVP；原 v0.1.0 与 B300 交接版的训练代码相同，因此只保留一份源码。

进入 `gopd-meta-teaching-mvp/` 后再执行训练命令。`README_MVP_zh.md` 解释方法与配置，`HANDOFF_B300.md` 给出 B300 执行顺序。`README.md` 保留 G-OPD 上游说明与来源。

验证记录只覆盖已完成的 CPU 与接口检查；真实 B300 训练待执行端验收。模型、训练数据、运行日志和 checkpoint 留在训练机器上。

[返回主目录](../README.md) · [阅读笔记](../reading-notes/README.md)
