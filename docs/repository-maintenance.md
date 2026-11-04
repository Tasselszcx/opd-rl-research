# 仓库维护记录

仓库从论文阅读集合逐步扩展到研究代码，因此改名为`opd-rl-research`。本次保留原Git历史，在独立整理分支中完成目录和README更新。

| 原位置 | 当前位置 | 处理 |
|---|---|---|
| `reading-notes/` | 保持原位 | 论文、PDF、离线资源保留，仅更新仓库导航 |
| `repositories/gopd-meta-teaching-mvp/` | `archive/gopd-meta-teaching-mvp/` | 原型源码、许可证和验证记录保留，增加历史状态说明 |
| 旧根README的部署指令 | `archive/`中的原文 | 根README改为当前研究导航，不再默认启动旧原型 |
| 当前训练源码快照 | `experiments/student-state-guidance/` | 新增G3a、配置、输入校验、导出工具与测试 |
| 本地机器地址和运行路径 | 不发布 | 由运行者通过环境变量设置 |

GitHub仓库链接与在线阅读入口均已改用新名字。GitHub Pages的项目路径会随仓库改名而变，旧项目站点URL不自动重定向，使用README中的新入口。参考[GitHub改名说明](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository)。

## 后续更新

- 研究设计以当前G3a配置为准。旧原型和旧全矩阵只作为历史资料，不自动恢复。
- 结果快照要带采集时间、任务、学生起点、教法、训练种子与验证步，开发结果和正式终测分开。
- `results/g3a/`只放精简表格、图与来源哈希；原始回答、checkpoint、模型、环境和私有机器配置留在运行存储中。
- 不改写失败或中断成本，不删除较差条件，不用挑选过的checkpoint冒充固定终点。
- 修改源码后运行CPU检查；修改笔记或导航后检查本地链接与离线资源。
