# task7 · hello-gpu Day5 项目验收套件

Task7「项目验收与修复」（deepagents-in-action 第 13 章）的落地：把「固定输入 + 明确预期 + 证据」用到 Day5
kernel_optimize Agent 项目上，验收它的权威裁判（chapter14/evaluate.py）。
本文件只讲怎么复现；**打卡报告在 `../Task7_打卡笔记.md`**。

仓库内位置：套件在本目录（`courses/09-deepagents-in-action/task7/`），证据在
`courses/09-deepagents-in-action/evidence/task7/`，被测项目（裁判）在
`courses/01-hello-gpu/code/day5_agent/tutorial/code/part3-agent/`。

## 目录结构

```text
courses/09-deepagents-in-action/
├── Task7_打卡笔记.md          # 打卡报告（验证什么/改了什么/实际结果/证据与解释/下一步修正）
├── task7/                    # 验收套件（本目录）
│   ├── README.md             # 本文件
│   ├── cases/                # 固定输入 case：预期在跑之前写死在这里
│   │   ├── caseA_accept_baseline.json   # 裁判必须接受正确 baseline
│   │   ├── caseB_accept_improved.json   # 裁判必须接受单变量候选并给出配对裁决
│   │   └── caseC_reject_wrong.json      # 裁判必须拒绝算错的候选
│   ├── candidates/           # 固定输入本体（task.json/reference.py 逐字节复制自 fixture）
│   │   ├── task.json  reference.py  baseline.py
│   │   ├── optimized_block1024.py    # 唯一改动：block_size 256→1024
│   │   └── wrong_multiply.py         # x*y 冒充 x+y（过源码政策、不过正确性）
│   ├── run_acceptance.py     # 运行器：核对输入身份→调项目裁判→对账预期→落证据
│   ├── rubric_gate.py        # 最小 Rubric 门：6 条全过才放行，失败 exit 1
│   ├── frozen_inputs.json    # 固定输入 sha256 冻结值
│   └── run_full.log          # 整套验收未裁剪日志
└── evidence/task7/           # 证据：裁判原始 JSON + 终端输出 + manifest + 扫描 + 门记录
```

## 环境与依赖版本（实测环境）

| 组件 | 版本 |
|---|---|
| GPU | NVIDIA RTX 4090 D 24GB（vm-60，Ubuntu 22.04，Linux 6.8） |
| Python | 3.10.12 |
| torch | 2.9.1+cu128 |
| triton | 3.5.1 |
| 裁判声明版本（pyproject.toml） | torch 2.10.0+rocm7.12.0 / triton 3.6.0+rocm7.12.0 / Python≥3.12（ROCm） |

注意：裁判以 `python -I` + `PYTHONNOUSERSITE=1` + 全新 HOME 运行 worker 子进程，
**torch 装在 user site 的机器上子进程看不见**。本次运行机器（vm-60）的解法（详见 ../Task7_打卡笔记.md 问题 1/2）：

```bash
python3 -m venv --without-pip --system-site-packages ~/task7-acceptance/venv
SITE=$(~/task7-acceptance/venv/bin/python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
USER_SITE=~/.local/lib/python3.10/site-packages
# 1) 软链 torch/triton 及其导入链依赖（按报错迭代：torch, typing_extensions, torchgen, numpy, triton）
# 2) 软链全部 dist-info/egg-info（triton 靠 importlib.metadata 的 entry points 发现后端）
for meta in "$USER_SITE"/*.dist-info "$USER_SITE"/*.egg-info; do ln -sfn "$meta" "$SITE/$(basename "$meta")"; done
```

## 启动步骤（复现整套验收）

```bash
# 0. 准备（vm-60 运行时的目录布局；仓库中同一套文件在
#    courses/09-deepagents-in-action/task7/，被测项目在
#    courses/01-hello-gpu/code/day5_agent/tutorial/code/part3-agent/）：
#    ~/task7-acceptance/part3-agent/        ← tutorial/code/part3-agent（逐字节复制）
#    ~/task7-acceptance/task7-acceptance/   ← 本套件（task7/ 的内容）

# 1. 进入套件目录，用能看见 torch+triton 的解释器跑验收（含单变量扫描）
cd ~/task7-acceptance/task7-acceptance
~/task7-acceptance/venv/bin/python run_acceptance.py \
    --part-root ../part3-agent --out evidence --sweep --freeze

# 2. 过 Rubric 门（6 条全过 exit 0；任何一条不过 exit 1）
~/task7-acceptance/venv/bin/python rubric_gate.py --evidence evidence

# 3. 负向自测：证明门会关闭（临时副本上篡改证据，不动真证据）
~/task7-acceptance/venv/bin/python rubric_gate.py --evidence evidence --selftest-negative
```

预期输出：3/3 case 符合冻结预期；扫描 5 个数据点全部 ok；门 6/6 PASS（verdict: ACCEPTED）；
负向自测报告 `the gate closes on failure`。任何一步不符，先看 `evidence/task7/*.terminal.txt`
与 `run_full.log` 的未裁剪输出，**不改输入和评测规则强行出加速**。

## 模板提交说明

- 证据规范遵循 `courses/01-hello-gpu/evidence/README.md`：真实运行产物、未裁剪终端输出、
  同一任务契约与计时协议下的对比。
- 提交物 = `../Task7_打卡笔记.md` + `../evidence/task7/`（+ 本目录的 README 与两个脚本，保证可复现）。
- 固定输入身份（sha256）记录于 `frozen_inputs.json` 与 `evidence/manifest.json`，可独立复核。
