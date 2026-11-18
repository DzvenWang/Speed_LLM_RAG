# ReShift 论文初稿（TR-C，elsarticle v3.4 模板）

- `main.tex`：主文件（前言、摘要、亮点、关键词，按章节 `\input`）
- `sections/`：introduction / related_work / problem / method / experiments / results / discussion / conclusion
- `references.bib`：参考文献（`stlaformer`、`traffident` 是占位条目，必须替换；其余页码投稿前再核对一遍）
- `elsarticle.cls`、`elsarticle-harv.bst`：模板文件（作者-年份引用格式）

编译（输出在 `build/main.pdf`）：

    mkdir -p build && pdflatex -output-directory=build main && (cd build && BIBINPUTS=.. BSTINPUTS=.. bibtex main) && pdflatex -output-directory=build main && pdflatex -output-directory=build main

红色的 `--` 和 `[TODO: ...]` 是待填项。结果章节里的所有数字都要等 ReShift-STLA v0.2 正式实验跑完后再填。
