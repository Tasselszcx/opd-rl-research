# 本次发布验证

当前运行核心对应`0ee2ff2bebe692fdc5d428db5ae7cb25551ae2c4`。公开副本保留可移植路径和输入准备入口；活跃训练release独立保存。

- Linux隔离副本的126项CPU测试全部通过，覆盖结束符语义对齐、换行编码、teacher-only响应保留、存储、调度、身份和准备入口。
- 用两个独立Python进程比较公开副本与运行核心：1条参考＋8条分支，共9份生成配置，给定相同显式路径时全部一致。
- 仓库检查核对731个源码快照文件、导航链接、发布体积以及凭据/私有路径规则。原有阅读笔记与许可保留。

独立GitHub CI发现公开依赖清单漏列`math-verify`，现已补齐运行环境中的0.8.0及其LaTeX解析依赖。判分测试保留，不以跳过测试替代依赖修复。[持续检查](https://github.com/Tasselszcx/opd-rl-research/actions/workflows/checks.yml)

真实GPU证据来自运行核心：原配置25步回归通过，随后两条正式分支已完成真实首批更新。发布路径适配没有额外做GPU采样；CPU通过不能代替八组150步终点和独立终测。

换行转义错误来自本项目启动封装。旧EOS屏蔽沿用上游，用于避免师生结束符不一致；当前先对齐结束语义，再保留结束监督。未修改原评分规则。

精确重放当前部署还需要未公开的历史30/90完整存档；公开从头准备得到的是新起点。原始模型、数据、指导预期SHA保留，完整模型SHA由bootstrap缓存步骤核验，不在每次启动时重复重算。

[验证摘要](../results/g3a/validation.json) · [来源与公开适配](../experiments/student-state-guidance/PUBLICATION_PROVENANCE.json) · [旧发布记录](../results/g3a/legacy-20261104/validation.json)
