# 检查范围与证据

当前运行核心为`5c4cfa4cef25cb31002811bfbecf39165b2dda32`。clean-v4从官方Base重新训练公共参考，旧存档不再作为新起点。

- 32项相关CPU检查通过，覆盖新起点来源限制、开发种子下传、EOS、真实换行、存储与范围。
- 现有运行已验证首个真实更新：从step0开始、无checkpoint恢复，128题匹配seed42，teacher信号、响应对齐、EOS和梯度检查通过。
- 全量输入审计：21,664道原题最多361 token，20,000份完整指导最多1616，均低于4096输入上限。Base与teacher的当前非思考模板渲染一致。
- 旧轮77,824条输出评分复算一致，按首个特殊结束符截断后评分也不变。输出仍可能触及4096预算上限；这类回答照原规则计分，不能说所有模型回答都已完善。

公开准备入口同时写入新起点约束，并与当前源码保持科学配置一致。公开路径适配的17项针对性CPU检查与9份配置比较见[验证摘要](../results/g3a/validation.json)；未为发布重复做GPU采样。[源码与公开适配](../experiments/student-state-guidance/PUBLICATION_PROVENANCE.json)

旧版126项发布测试及历史诊断结果单独保留在[旧发布验证](../results/g3a/legacy-format-v3-20261105/validation.json)。当前CPU和首步通过，不代表八组150步终点、独立终测或状态交互已经完成。

[GitHub持续检查](https://github.com/Tasselszcx/opd-rl-research/actions/workflows/checks.yml) · [当前进度](g3a-results.md)
