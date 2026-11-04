# 来源与许可

仓库包含研究笔记和多个代码来源，不给整仓库套用一个覆盖所有论文、数据和第三方资产的统一许可。

| 内容 | 来源与许可入口 |
|---|---|
| 当前G3a训练实现 | [StringNLPLAB/opd-rlvr](https://github.com/StringNLPLAB/opd-rlvr)，原 [LICENSE](experiments/student-state-guidance/LICENSE) 与 [NOTICE](experiments/student-state-guidance/NOTICE)保留 |
| vendored verl | [verl](https://github.com/volcengine/verl)，[LICENSE](experiments/student-state-guidance/verl/LICENSE) |
| reasoning-gym | [reasoning-gym](https://github.com/open-thought/reasoning-gym)，[LICENSE](experiments/student-state-guidance/logic/LICENSE.reasoning-gym)与[NOTICE](experiments/student-state-guidance/logic/NOTICE.txt) |
| 数学评测代码 | [DeepMath](https://github.com/zwhe99/DeepMath)，[LICENSE](experiments/student-state-guidance/math/LICENSE.deepmath) |
| 历史G-OPD原型 | [MVP_UPSTREAM.json](archive/gopd-meta-teaching-mvp/MVP_UPSTREAM.json)列明MIT/Apache来源及修改范围 |
| 论文、PDF、网页存档 | 保留各原作者的版权；引用出处见阅读页面与各论文链接 |
| 字体、KaTeX、Mermaid | [阅读页面资源许可](reading-notes/README.md) |

Qwen训练权重与正式训练/评测数据不随本代码分发，使用时遵守各发布方许可。vendored verl保留上游自带的约1.6MB算术序列玩具测试模型，它不是本实验训练的权重，原始SHA由来源清单核验。源码发布的固定版本、保留文件和路径适配记录见 [PUBLICATION_PROVENANCE.json](experiments/student-state-guidance/PUBLICATION_PROVENANCE.json)。
