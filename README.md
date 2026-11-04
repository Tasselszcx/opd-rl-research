# OPD / RL 研究与实验

这里整理论文阅读、复现代码和我们自己的实验。当前主要工作是 **G3a：固定老师，比较学生学到不同阶段时，哪种指导更有用**。

| 想看什么 | 从这里开始 |
|---|---|
| 当前实验想验证什么、怎么比较 | [G3a实验说明](docs/g3a-study.md) |
| 当前训练、评测和恢复代码 | [student-state-guidance](experiments/student-state-guidance/README.md) |
| 已保存的开发集结果与曲线 | [结果快照](docs/g3a-results.md) |
| 论文中文解读、原文和离线页面 | [阅读笔记](reading-notes/README.md) · [在线阅读](https://tasselszcx.github.io/opd-rl-research/reading-notes/) |
| 早期G-OPD元教学原型 | [历史归档](archive/README.md) |

## 当前实验

只做K&K逻辑题和GSM8K数学题，只跑训练种子42。先准备训练30步和90步的学生，再分别比较不给提示、只给答案、完整解法、针对错误给提示四种教法。每条对比都继续训练150步，共 **2条准备训练＋16条对比＝18条**。

老师固定为Qwen3-8B，学生为Qwen3-1.7B-Base。提示只给老师看，学生始终只看原题。本轮只做OPD，不训练指导生成器，也不进入独立RL阶段。

结果文件是带日期的快照，不是实时看板。开发集成绩与最后的独立测试分开；实验未完成时，不把单个条件的变化写成“最佳教法”或状态差异结论。

## 目录

```text
experiments/student-state-guidance/  当前G3a代码、配置、测试和运行入口
docs/                               实验说明、结果解释、验证与维护记录
results/g3a/                        精简开发集表格和图；不含原始回答或权重
reading-notes/                      论文阅读页面、原文PDF及生成工具
archive/gopd-meta-teaching-mvp/      早期原型，保留来源与原有验证记录
tools/                              仓库结构、链接与发布内容检查
```

代码与数据准备步骤见[当前代码README](experiments/student-state-guidance/README.md)。Qwen训练权重、运行环境、真实checkpoint、原始训练/评测输出和机器配置均不存进Git；vendored代码仅保留上游自带的小型玩具测试模型。

来源与许可见[第三方说明](THIRD_PARTY_NOTICES.md)。旧目录的迁移关系见[维护记录](docs/repository-maintenance.md)。
