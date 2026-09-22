# TinyTorch Task 3 自动微分引擎 — 一键复现脚本（Windows / PowerShell）
# 用法（在 hello-gpu 工作区根目录）:
#   .\courses\07-tinytorch\code\task3\task3_verify.ps1
# 依赖:  .venv-task3（numpy + pytest + Pillow，版本见日志）
# 说明:  不用 $PSScriptRoot（DSH  harness 下该变量指向用户主目录，不可靠），改用当前工作目录定位。
# 注意: 不用 "Stop" —— 原生命令(pytest)往 stderr 写提示时, PS 5.1 会抛终止性错误
$ErrorActionPreference = "Continue"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$Root    = (Get-Location).Path                                    # hello-gpu 工作区根（需含 _tmp_tt_src 与 courses）
if (-not (Test-Path (Join-Path $Root "_tmp_tt_src\tinytorch\src\06_autograd\06_autograd.py"))) {
    throw "请在 hello-gpu 工作区根目录运行（找不到 _tmp_tt_src\tinytorch）"
}
$Repo    = (Resolve-Path (Join-Path $Root "_tmp_tt_src\tinytorch")).Path   # tinytorch 源码仓
$CodeDir = (Resolve-Path (Join-Path $Root "courses\07-tinytorch\code\task3")).Path
$EvidDir = Join-Path $Root "courses\07-tinytorch\evidence\task3"
$Python  = (Resolve-Path (Join-Path $Root ".venv-task3\Scripts\python.exe")).Path

New-Item -ItemType Directory -Force -Path $EvidDir | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $Repo
Set-Location $Repo

Write-Host "[1/6] 环境信息"
& $Python -c "import sys, numpy, pytest; print('Python', sys.version.split()[0]); print('NumPy', numpy.__version__); print('pytest', pytest.__version__)" | Tee-Object -FilePath (Join-Path $EvidDir "task3_env.txt")

Write-Host "[2/6] 导出 Module 06 -> tinytorch/core/autograd.py"
& $Python (Join-Path $CodeDir "export_autograd.py") | Tee-Object -FilePath (Join-Path $EvidDir "task3_export.log")

Write-Host "[3/6] 模块自测 test_module()"
& $Python (Join-Path $CodeDir "run_selftest.py") 06 | Tee-Object -FilePath (Join-Path $EvidDir "task3_06_autograd_selftest.txt")

Write-Host "[4/6] pytest tests/06_autograd"
& $Python -m pytest tests/06_autograd -v --tb=short 2>&1 | Tee-Object -FilePath (Join-Path $EvidDir "task3_pytest_06.log")

Write-Host "[5/6] pytest tests/01..06 全量回归"
& $Python -m pytest tests/01_tensor tests/02_activations tests/03_layers tests/04_losses tests/05_dataloader tests/06_autograd -q --tb=short 2>&1 | Tee-Object -FilePath (Join-Path $EvidDir "task3_pytest_full_01_06.log")

Write-Host "[6/7] autograd 引擎证据脚本"
& $Python (Join-Path $CodeDir "demo_autograd.py") | Tee-Object -FilePath (Join-Path $EvidDir "task3_autograd_demo.txt")

Write-Host "[7/7] 逐文件 pytest 汇总 + 渲染打卡截图"
& $Python (Join-Path $CodeDir "pytest_by_file.py")
$Title = "从Tensor到Transformer——构建 GPT 模型 | Task 3 自动微分引擎"
$Meta  = "GitHub ID: wsm000  |  微信群昵称: empty  |  本地 Windows / Python 3.10.2 / NumPy 2.2.6 / pytest 9.1.1"
$Cwd   = "C:\...\_tmp_tt_src\tinytorch"
$Shot   = Join-Path $CodeDir "make_runshot.py"
& $Python $Shot (Join-Path $EvidDir "task3_06_autograd_selftest.txt") (Join-Path $EvidDir "task3_06_autograd_runshot.png") $Title $Cwd "python run_selftest.py 06" $Meta
& $Python $Shot (Join-Path $EvidDir "task3_pytest_by_file.log")     (Join-Path $EvidDir "task3_pytest_runshot.png")     $Title $Cwd "python -m pytest tests/06_autograd -q --tb=short" $Meta
& $Python $Shot (Join-Path $EvidDir "task3_autograd_demo.txt")      (Join-Path $EvidDir "task3_demo_runshot.png")        $Title $Cwd "python demo_autograd.py" $Meta

# 规范化: PowerShell Tee-Object 默认写 UTF-16, 统一转成 UTF-8 便于入库
Get-ChildItem $EvidDir -File | ForEach-Object {
    $b = [System.IO.File]::ReadAllBytes($_.FullName)
    if ($b.Length -ge 2 -and $b[0] -eq 0xFF -and $b[1] -eq 0xFE) {
        $t = [System.Text.Encoding]::Unicode.GetString($b)
        [System.IO.File]::WriteAllText($_.FullName, $t, (New-Object System.Text.UTF8Encoding $false))
        Write-Host ("normalized to UTF-8: " + $_.Name)
    }
}

Write-Host "完成，证据目录: $EvidDir"
