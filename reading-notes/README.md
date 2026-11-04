# OPD 与 RLVR 文献调研

11 篇论文的中文方法解读、训练步骤、实验结果与对应原文 PDF。

**[在线阅读](https://tasselszcx.github.io/opd-rl-research/reading-notes/)**

也可以下载仓库，解压后用浏览器打开 `reading-notes/index.html`。页面、公式、流程图和 PDF 支持离线阅读。

## 论文

1. RISE
2. SOAR
3. Learning from the Near Future：NPO / NPD / AutoNPO
4. Sequential Beats Joint
5. OPW
6. MetaOPD
7. PEAR
8. Dr. OPD
9. OP-Short
10. RL Starts before RL
11. ATESD

每篇页面顶部提供论文链接及本地 PDF，文末提供正文依据和公开代码来源。实验数据引自论文。

## 文件

- `index.html`：阅读入口。
- `原文PDF/`：11 篇论文原文。
- `data/papers.json`：内容与顺序的唯一数据源。
- `scripts/`：生成及检查工具。
- `assets/`：本地字体、公式和流程图资源。
- `sources/`：原文存档及引用信息。

## 更新页面

```bash
cd reading-notes
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/build.py
```

修改 `data/papers.json` 后运行生成脚本，再提交生成的页面即可。GitHub Pages 从 `main` 分支根目录发布，阅读内容位于 `reading-notes/`。

论文和第三方资源保留各自的版权及许可证，相关许可证位于 `assets/fonts/OFL.txt` 和 `assets/licenses/`。

研究方案与文献调研见 [OPD warmup 元教学调研](OPD_warmup_meta_teaching_论文调研与实验建议.md)。当前实验代码见 [G3a](../experiments/student-state-guidance/README.md)，早期原型见 [历史归档](../archive/README.md)。
