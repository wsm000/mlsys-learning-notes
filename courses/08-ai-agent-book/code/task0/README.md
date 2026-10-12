# Task0 MiMo 适配入口

这是本仓库的适配脚本，直接导入上游课程 `ContextAwareAgent`。仅在运行时注册 AMD 提供商，两组上下文处理沿用课程代码。环境变量中的密钥不写入结果文件。

先克隆 [课程仓库](https://github.com/bojieli/ai-agent-book)，记录 `git rev-parse HEAD`，再创建独立 Python 环境并安装本目录的 `requirements.txt`。不要将密钥写入脚本或提交到 Git。

在 PowerShell 中设置课程目录，并以隐藏输入方式临时提供密钥：

```powershell
$env:AI_AGENT_BOOK_DIR = 'D:/repos/ai-agent-book'  # 改为你的课程目录
$taskSecure = Read-Host 'AMD API Key' -AsSecureString
$taskPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
try {
  $env:AMD_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPtr)
  $env:PYTHONUTF8 = '1'
  python run_context_mimo.py
} finally {
  Remove-Item Env:AMD_API_KEY -ErrorAction SilentlyContinue
  [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPtr)
}
```

默认结果写入课程目录的 `evidence/local-context-run/`，避免覆盖本次已记录的证据。可用 `AGENT_RESULTS_DIR` 指定其他目录。它运行 full 与 no_tool_results，每组最多 5 轮；不要把 `completed` 单独当成答案正确的判断。

课程工具包括 Python 代码执行，仅在个人实验环境使用。当前课程 main 分支可能继续变化；本次使用源码的哈希见 evidence/task0/source-manifest.json。
