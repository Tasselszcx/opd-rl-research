# K&K 指导方式小试验 G1

2026-10-23，用户选择“指导方式小试验”，并负责释放新机器的四卡占用。

## 问题与对照

从相同 Qwen3-1.7B-Base student 出发，固定 Qwen3-8B teacher，比较三种仅提供给 teacher 的参考信息：

1. `plain`：原题，无附加参考信息。
2. `key_step`：由精确布尔求解器证明的一个中间结论。按最少充分语句子集、固定字典顺序选择一个被强制确定的角色；不披露完整答案。
3. `full_solution`：完整可验证的枚举推理，记录依次施加语句约束后的剩余解数、唯一完整角色分配及一致性检查。

student 始终只接收原题；其生成 token 保持原样送入 teacher 条件评分。teacher 的参考信息改变 token 级蒸馏分布，不是把 teacher 文字意见当作奖励。每种方式做 20 个 OPD batch，再从真实 checkpoint 接 20 个 GRPO batch，到全局 40。RL 阶段撤掉指导信息，目标中不使用 teacher 信号；保留作者代码未用于目标的 teacher 前向并计入成本。

本轮不训练指导生成器、不做在线教法选择、不增加训练种子或搜索超参数。它测试控制教法是否改变即时学习和随后 RL 的最终表现；不是完整元教学方法，也不是重新复现原论文 150 步设置。

## 固定设置与预算

- 复用作者 StringNLPLAB/opd-rlvr 基线及已验证的运行补丁，另开代码目录，原实验不变。
- 三组使用同一 Base 模型、同一训练种子和题目顺序；每组四张 GPU，顺序运行。
- 小试验 batch=32、rollout=4、mini-batch=16、每卡 micro-batch=1、response=2048、prompt=1024、lr=1e-6；训练采样保持作者 temperature=1、top-p=1、top-k=-1。正式启动前逐题检查20,000题三种teacher输入都能放入1024，不截断；比原计划4096减少无效padding。此项在打开新测试分数之前按输入长度确定，记录prompt_length_audit.json。
- 固定训练池沿用原 20,000 题；新 meta_val 为128题（composite seed=2000000），新 pilot_test 为512题（seed=3000000），三者逐题 hash 验证零交集。
- 每组仅在固定 OPD20 和 RL40 两个 checkpoint 上最终测 pilot_test，每题4次采样，temperature=0.7、top-p=0.8、top-k=20、seeds42–45、response=2048。
- 三种方式的两个阶段全部结束后统一打开最终测试；meta_val 仅记录进展，不用于选步数、挑 checkpoint 或改配置。
- **本轮总上限24 GPU小时**，包含当前迁移训练链路检查、后续指导运行检查、失败尝试、训练、评测；CPU数据准备单独记录。不自动扩预算。达到上限时保留进度并报告未完成，不将不同长度的残缺分支当完整比较。
- 先通过限定的一步OPD→一步RL迁移检查，再做有指导输入的运行检查。工程检查结果不计作研究分数。

## 必须核验

- teacher-only输入确实不同，student prompt/response token及其mask完全不变；禁止teacher提示泄漏给student。
- 指导只来自训练题oracle；key_step确为逻辑蕴含，full_solution唯一解与原答案一致。
- checkpoint恢复后batch顺序接续；三组阶段长度一致，RL阶段teacher目标计数为0。
- 保存指导内容、数据哈希、启动参数、版本、逐题输出、吞吐、teacher token/序列数、显存、真实GPU成本。

## 报告

分别报告OPD后和RL后的独立集pass@1/pass@4、三组曲线和相对plain的题目级配对bootstrap区间。关键比较是最终表现相对同起点plain的差异，不单独用“RL后减OPD后”给教法排名。指出即时OPD排序是否与RL后排序一致，但只作单训练种子的小试验结论。
