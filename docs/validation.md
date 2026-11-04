# 发布验证

本次整理使用固定训练源码快照`140f0cb5935637c7ccc500240da9c786056424cc`，在独立副本中进行。没有修改运行中的release，没有重启训练，也没有为仓库整理新增GPU采样。

已完成的代码检查：

- Linux参考环境中运行 **95项CPU测试，全部通过**，包括原有协议、teacher-only响应对齐、优化器/采样恢复、进程身份、看护、范围与终测等待测试，以及7项公开准备工具检查。
- 将2条参考与16条正式分支共 **18份生成配置** 与原运行代码逐项比较：给定相同显式运行根和模型目录时，科学训练设置一致。
- 模型原始文件、数据划分与指导的预期SHA随配置保留；公开协议副本只移除部署路径，范围中的协议SHA相应重新计算，原协议SHA在发布来源中保留。
- 原有阅读内容与PDF保留。仓库结构、链接和发布内容检查由`tools/check_repository.py`执行。

训练主体的真实GPU运行记录来自原部署；**本次整理版没有重新做GPU训练验收**。CPU通过不等于全部18条完成，也不等于独立终测已完成。最终训练与统计结论以固定终点的完整评测为准。

模型验证分两步：`bootstrap_cache`重算全部23项SHA；后续`public_setup check`检查已有验证记录和文件大小，只有显式传`--hash-models`才重新计算权重SHA。版本列表不是驱动/容器锁。

发布的精简验证记录见 [validation.json](../results/g3a/validation.json)，代码来源与公开适配见 [PUBLICATION_PROVENANCE.json](../experiments/student-state-guidance/PUBLICATION_PROVENANCE.json)。
