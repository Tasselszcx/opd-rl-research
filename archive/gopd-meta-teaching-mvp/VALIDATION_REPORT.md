# MVP 验证范围
## 已完成
- 完整 CPU 测试环境：**137 passed，0 failed**；1 条上游 profiler 配置警告。
- 新建轻量 CPU 环境：**95 passed，6 skipped，0 failed**；缺少训练依赖的测试明确跳过。
- Python 编译检查、两份 Bash 脚本语法检查、Git diff whitespace 检查通过。
- 四份 JSON 配置可加载，生成的 warmup/RL/eval 命令通过真实 Hydra 和上游 dataclass 校验。
- 真实 Parquet → RLHFDataset → collate → DataProto → 教师输入链路通过；训练参考解不进入学生输入，meta_val/test 不带参考解。
- 用真实 PyTorch 小模型和 AdamW 验证：教师信号会改变学生梯度；跳过的样本不提供 OPD 梯度；RL 阶段的更新不依赖教师分数。
- 阶段恢复、教师退出条件、坏摘要拒绝、超时/中断/重试费用、数据替换拒绝与配对分支逻辑均有测试。
- 使用真实轻量子进程测试启动器和导出器生命周期；子进程中的模型文件和分数仅为 fixture。
## 尚未验证
**没有在 GPU 上运行这份 MVP。** CUDA ABI、NCCL、FSDP 八卡启动、vLLM 生成、显存峰值和模型收敛，需要在目标机器执行 smoke。
默认依赖面向 Ampere/Hopper。B300/B200 不能使用旧 Hopper 安装 profile，应使用另外验证过的兼容环境。
模拟流程只验证调度和记录，模拟分数不代表模型性能；本包没有真实训练结果或模型权重。
## 如何复查
在工程根目录运行：
```bash
python -m pytest tests_mvp -q
```
轻量安装使用 mvp_requirements_cpu.txt；有训练依赖的环境可覆盖更多集成测试。原始摘要见 validation/full-suite.txt、validation/lean-suite.txt；结构化范围见 validation/report.json。
## 八卡 smoke 验收
1. preflight --gpu 全部必需检查通过。
2. warmup 的 teacher_queries 大于零；RL/eval 的教师计数全为零。
3. 每个 completed 阶段满足 end_step-start_step=phase_steps，且 final validation_step=end_step。
4. checkpoint 包含全部八个 rank 的 model、optimizer、extra、rollout RNG 和 dataloader。
5. 实验能完成至少一轮，产生 branches.csv、teaching_policy.json 和真实 results.json。
6. 原样 --resume 不重复启动已完成阶段；任何数据内容变化必须拒绝复用。
7. 只有通过上述验收后，再运行 pilot/budget 配置。正式效果比较仍需未见任务与独立训练种子。
