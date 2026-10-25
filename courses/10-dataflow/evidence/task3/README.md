# Task3 证据索引

| 文件 | 内容 |
|---|---|
| dataflow-task3-checkin.md | 打卡文案 |
| runshot_task3_mineru_layout_math1.png / _math2.png | MinerU 版面识别可视化（题号框/表格/插图/公式） |
| mineru_math1.md / mineru_math2.md | 解析后的 Markdown（公式 LaTeX、表格 HTML、图题） |
| mineru_math1_content_list.json / mineru_math2_content_list.json | 结构化版面条目（含 bbox、page_idx、type） |
| mineru_*_layout.pdf | MinerU 自带版面可视化 PDF |
| mineru_smoke_* | 4 页扫描 PDF 冒烟测试产出 |
| smoke_mineru.log | 冒烟日志（rc=0，52.1s，40 items） |
| install_df3.log / verify_df3d.log | 环境安装与 import 验收 |
| task3_pipeline.log | Case 7 全链路运行日志 |
| task3_verify.log | 验收结果（22/22 通过） |
| llm_raw_math1.txt / llm_raw_math2.txt | qwen3:1.7b 抽取原始输出 |
| math1/2_extracted_vqa.jsonl | 解析后 QA（结构有效、内容错误） |
| math1/2_merged_qa_pairs.md | QA_Merger 输出 |
| fallback_math1_merged_qa_pairs.md | 降级解析器输出（对照） |
| qa_sharegpt.json / fallback_qa_sharegpt.json | ShareGPT 最终产物 |
| case6_ocr_results.json | moondream VQA 结果 |
| prompt_exp.log / prompt_exp2.log | "只输出 id" 强调实验 |
| vqa_images/ | QA 引用的教材插图 |
