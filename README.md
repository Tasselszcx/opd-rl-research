# OPD / RL 研究与实验

当前实验：**K&K学生处于不同训练阶段时，哪种老师指导更有用？** 老师固定Qwen3-8B，学生Qwen3-1.7B-Base；学生自己回答，老师提供逐token概率，只更新学生。

本轮已切换到**clean-v4全链路修复版**：从官方Base重新做公共参考训练，保存新的30／90步完整状态，再比较四种指导。共**8条正式分支，各追加150步**；旧参考存档不再复用。

| 内容 | 入口 |
|---|---|
| 目标、八组配置与判分 | [实验说明](docs/g3a-study.md) |
| 当前进度与新旧结果边界 | [结果快照](docs/g3a-results.md) |
| 代码、输入准备与启动 | [代码README](experiments/student-state-guidance/README.md) |
| 论文中文解读与原文 | [阅读笔记](reading-notes/README.md) · [在线阅读](https://tasselszcx.github.io/opd-rl-research/reading-notes/) |
| 旧起点探索与早期原型 | [旧轮结果](results/g3a/legacy-format-v3-20261105/README.md) · [历史代码](archive/README.md) |

新公共参考已验证完成首个真实更新；正式8组等待新起点。旧轮两组完成150步，另两组按用户要求停止，结果和成本保留为历史探索，不能算作新轮进度。当前独立终测尚未开始。

EOS监督、真实换行和开发采样种子修复从参考训练第1步生效。[检查范围](docs/validation.md)。权重、原始回答、凭据和机器配置不进Git；许可见[第三方说明](THIRD_PARTY_NOTICES.md)。
