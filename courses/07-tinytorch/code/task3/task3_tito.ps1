# TinyTorch Task 3 官方 CLI 完成流: tito module complete 06(幂等, 可重复运行)
# 说明: 本地工作副本是 src/*.py(jupytext percent) 变体, 官方 tito 的导出步骤需要
#       modules/<NN>/<name>.ipynb; 本脚本先用 jupytext 生成 notebook, 再按官网
#       "How you know it works" 的四步(单元测试 -> 导出 -> 集成测试 -> 进度记录)跑完。
$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$Root    = (Get-Location).Path
$Repo    = (Resolve-Path (Join-Path $Root '_tmp_tt_src\tinytorch')).Path
$CodeDir = (Resolve-Path (Join-Path $Root 'courses\07-tinytorch\code\task3')).Path
$EvidDir = Join-Path $Root 'courses\07-tinytorch\evidence\task3'
$Python  = (Resolve-Path (Join-Path $Root '.venv-task3\Scripts\python.exe')).Path
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = $Repo
Set-Location $Repo

Write-Host '[1/5] 安装官方 CLI 依赖(rich/certifi/pyyaml/requests/jupytext/nbdev)'
& $Python -m pip install rich certifi pyyaml requests jupytext nbdev -i https://pypi.tuna.tsinghua.edu.cn/simple | Out-Null

Write-Host '[2/5] src/*.py -> modules/*.ipynb(jupytext)'
& $Python (Join-Path $CodeDir 'make_notebooks.py') $Repo

Write-Host '[3/5] tito module complete 01..06'
$log = Join-Path $EvidDir 'task3_tito_complete.log'
$all = foreach ($m in @('01','02','03','04','05','06')) {
    Write-Output ('===== tito module complete ' + $m + ' =====')
    & $Python bin/tito module complete $m 2>&1 | Out-String -Width 160
}
$all | Set-Content -Path $log -Encoding utf8
Write-Output ('log: ' + $log)

Write-Host '[4/5] 抽取关键行 + 渲染打卡截图'
& $Python (Join-Path $CodeDir 'tito_keylines.py')
$Title = '从Tensor到Transformer——构建 GPT 模型 | Task 3 自动微分引擎'
$Meta  = 'GitHub ID: wsm000  |  微信群昵称: empty  |  本地 Windows / Python 3.10.2 / NumPy 2.2.6 / pytest 9.1.1'
$Cwd   = 'C:\...\_tmp_tt_src\tinytorch'
& $Python (Join-Path $CodeDir 'make_runshot.py') (Join-Path $EvidDir 'task3_tito_keylines.log') (Join-Path $EvidDir 'task3_tito_complete_runshot.png') $Title $Cwd 'tito module complete 01..06' $Meta

Write-Host '[5/5] 回归确认(06 单模块 + 01-06 全量)'
& $Python -m pytest tests/06_autograd -q --tb=short 2>&1 | Out-String -Width 160
& $Python -m pytest tests/01_tensor tests/02_activations tests/03_layers tests/04_losses tests/05_dataloader tests/06_autograd -q --tb=short 2>&1 | Out-String -Width 160
Write-Host '完成'