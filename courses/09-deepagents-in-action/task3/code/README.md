# 实验代码

原始实验运行于既有 Python 3.11 环境，依赖 deepagents 0.7.8、langchain 1.4.3、langchain-openai 1.6.7。requirements.txt 锁定这三个直接依赖，不是完整传递依赖锁文件。

此处脚本从原始脚本适配：密钥改从 QUANTY_API_KEY 读取，输出默认保存到 task3/runs，工具与模拟资料随附。模型、提示词和实验逻辑保持原设计。原始证据中的绝对路径为历史记录，不需要在本机复刻。

在 Python 3.11+ 环境安装依赖，并通过本地安全方式配置 QUANTY_API_KEY。QUANTY_BASE_URL 可选，默认沿用原 AMD 兼容接口。模型账户需能访问 DeepSeek-V4-Flash。不要将密钥写进代码或提交到仓库。

```bash
python -m pip install -r requirements.txt
python task3_advanced_ab.py
python task3_summary_todo.py
```

可用 TASK3_OUTPUT_DIR 指定输出根目录。每轮带时间戳，避免复用上一轮产物。运行会调用真实模型并产生账户用量。

对照结果生成后，可离线统计：

```bash
python task3_audit_ab.py ../runs/task3-ab-实际时间戳
```

摘要脚本的观测子类保持默认中间件名称以进行原位替换，观测方法调用父类实现，记录真实摘要与请求。未人为插入消息、Todo 或摘要。低阈值及重复资料用于机制实验，不代表生产配置。

基础实验保留原始证据与复核，不在此另造一个声称复现过的新版基础运行入口。文字语义仍需人工复核；脚本的机械通过不代表全文解释正确。
