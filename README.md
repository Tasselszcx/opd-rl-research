# OPD / RL 研究与实验

当前实验是 **K&K：学生训练到不同阶段，老师该给哪种指导？**

老师固定为Qwen3-8B，学生为Qwen3-1.7B-Base。学生自己答题，老师对它的回答逐token打分；只更新学生。额外答案和解法只给老师看。

| 内容 | 入口 |
|---|---|
| 目标、八组对照与配置 | [实验说明](docs/g3a-study.md) |
| 修复结果与正式进度 | [结果快照](docs/g3a-results.md) |
| 代码、数据准备与启动 | [代码README](experiments/student-state-guidance/README.md) |
| 论文中文解读与原文 | [阅读笔记](reading-notes/README.md) · [在线阅读](https://tasselszcx.github.io/opd-rl-research/reading-notes/) |
| 早期原型与旧实验 | [历史归档](archive/README.md) |

当前只做K&K、seed42：四种指导×30/90两个学生起点，共**八条正式分支**，每条追加150步。另有一条共用的准备路线，提供起点；当前运行直接复用已保存的完整存档。

2026-11-05：结束符与提示换行处理已修复。原配置25步复验中，答案组正确率30.27%→36.06%，未触及长度上限却缺答案标签538/4096→0/4096。正式八组已进入队列，完整终测尚未完成。[详细口径](docs/g3a-results.md)

运行权重、原始回答、凭据和机器配置不进Git。来源与许可见[第三方说明](THIRD_PARTY_NOTICES.md)，目录变更见[维护记录](docs/repository-maintenance.md)。
