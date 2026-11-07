# 中文实验图

`scripts-zh/` 中的绘图入口将中文图件写入本目录。最终的
`Figure_*.pdf` 位于本层；中间 PDF、SVG、PNG、来源清单和派生表位于
`paper_artifacts/` 与 `semantic_analysis/` 子目录。

绘图脚本只读取 `results/` 中保留的实验结果。算法名称保持英文，其他
可见标题、坐标轴、场景和关系名称使用中文。
