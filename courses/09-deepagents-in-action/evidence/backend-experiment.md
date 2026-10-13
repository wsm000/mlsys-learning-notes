# StateBackend 写入与读回实验

2026-10-13，在vm-60课程主应用环境中调用build_agent()，使用DeepSeek-V4-Flash。

实际顺序：analyze_price_volume(window=3) → write_file(/reports/backend-b1bf51854c3f42f2b5c662fa64e05e18.md) → read_file(同一路径)。三个请求均有对应调用ID的成功返回。

写入内容与返回状态files中的内容完全一致。read_file返回包含涨幅2.91%、均量1300/1700、比值1.31及模拟数据说明。返回状态中存在该文件；运行前后检查远端绝对路径和应用目录下同名reports路径，都不存在磁盘文件。只检查这两个目标路径，不声称扫描过整个磁盘。

最初验收脚本误把字符串内容当作行列表拼接，且预期比值字符串错误，导致数值检查误报失败。已修正脚本，并对已捕获证据离线复验；原始误报保留在backend-experiment.json，修正结果在backend-verification.json。

backend-report.md由验证程序随后从状态中导出到本地，供阅读；这一步是我们导出的文件，不是Agent在vm-60写出的磁盘文件。未配置或验证跨运行持久化。

模型生成的报告中存在“performedcars”拼写错误，说明存储与读回正确不等于报告文案完美。
