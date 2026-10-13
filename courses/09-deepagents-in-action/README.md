# Quanty · Task1 量价研究助理

选做双模型对照已完成：[DeepSeek vs GLM](evidence/model-comparison-report.md)。各3次均2次完成、1次超时，成功返回的工具调用检查均通过，不能认定谁更稳定。

**最终材料入口：[Task1打卡笔记](Task1_打卡笔记.md)。** 第1–2章三份官方notebook已按锁定环境全部离线运行通过，51项官方测试通过。自建真实模型调用、课程主应用、边界对照、执行Trace和持久化实验均有证据。表单未提交；以下历史状态可能已被后续结果取代。

**2026-10-13最新：Advanced核心实验已有本地执行Trace。** [Trace报告](evidence/execution-trace-report.md)包含正常/超范围对照、调用层级与耗时；超范围仍多余调用ls，已记录效率缺陷。Lite/Basic材料已整理，未提交表单，未逐一运行课程全部示例。历史进度如下按时间保留，以本段为最新状态。

**当前进展：Lite笔记已整理，Basic主应用已跑通。** 见[打卡笔记](Task1_打卡笔记.md)与[组件定位](组件定位.md)。课程default模板当前为无前端的AG-UI网关；已在vm-60真实请求完成并记录内部工具事件。Advanced与Agent递归改进尚未实现。金融RSI历史实现已从活跃Agent移除。

## RSI 工具
新增 `calculate_rsi(symbol="DEMO", period=14)`，与量价比较工具并存，由模型选择。使用全部可用收盘价：先以 period 个涨跌初始化简单平均，再用 Wilder 平滑。完全不涨不跌约定为50（各库可能不同）；只有上涨为100，只有下跌为0。周期是涨跌次数，至少需要 period+1 个收盘价。

现有10条数据不足以计算14期，返回 `status=insufficient_data, rsi=null`；不自动缩短周期。可明确指定3期做教学实验。短历史的初始值影响较大，数据仍为模拟数据。

计算参考：[Fidelity RSI](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI)、[Fidelity Wilder平滑说明](https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/learning-center/trading-with-momentum-transcript.pdf)。

**最新：vm-60 上换为 DeepSeek-V4-Flash 已跑通真实调用闭环。** 同一AMD Key，1次业务工具请求对应1次成功返回；结果见 evidence/deepseek-success-trace.json。运行：`QUANTY_MODEL=DeepSeek-V4-Flash .venv/bin/python run_vm60.py`（vm-60项目目录内）。MiMo此前失败与原始排障过程保留如下。

最新执行位置：vm-60 的 /home/simin/course09-quanty。后续 Python 实验在该机展开。
远端 Python 3.11.15 / Deep Agents 0.7.24 已安装，3项工具测试通过；MiMo 两次真实请求分别返回504、502，尚未获得成功调用轨迹。见 evidence/vm60-progress.md。以下本地安装说明保留作参考。

课程9。沿用课程8的 AMD MiMo 接入：默认模型 MiMo-V2.6-Flash，默认接口 https://developer.amd.com.cn/radeon/api/v1，读取本地 AMD_API_KEY。当前进程未发现该密钥，尚不能执行真实模型请求。已保存工具结果 evidence/tool-result.json 和测试记录 evidence/tool-tests.txt；这些不是模型调用 Trace。

已创建 Python 3.12 独立环境。运行 Agent 前可在本目录的 PowerShell 安全输入课程8密钥：
```powershell
$taskSecure = Read-Host 'AMD API Key' -AsSecureString
$taskPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
try {
  $env:AMD_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPtr)
  .\.venv\Scripts\python agent.py
} finally {
  Remove-Item Env:AMD_API_KEY -ErrorAction SilentlyContinue
  [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPtr)
}
```

## 当前状态
已实现确定性工具与 Deep Agent 入口；模型调用尚未验证。DEMO 是手工构造的模拟数据，不是真实行情。先完成课程 Lite 调用闭环，之后再扩展真实数据与主应用。

## 第一步：亲自运行工具（无需第三方依赖）
在本目录运行：
```powershell
python tools.py
python -m unittest -v
```
默认比较最近5条与之前5条记录。涨幅按最后价格106 / 基准末日价格100 - 1 = 6%；日均成交量1600 / 1000 = 1.6。“放量”暂定义为比值大于1，没有统计显著性含义。

## 第二步：真正运行 Agent
使用 Python 3.11+ 创建独立环境（本机已发现的 Python 为3.10，不能直接当作兼容环境）。
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
# 在本地终端设置，勿提交密钥或发到聊天里
$env:QUANTY_MODEL = '你的模型名称'
$env:QUANTY_API_KEY = '你的密钥'
$env:QUANTY_BASE_URL = '你的OpenAI兼容接口地址' # 官方默认接口可省略
.\.venv\Scripts\python agent.py
```
模型必须支持工具调用。环境中的 LANGSMITH_TRACING、LANGSMITH_API_KEY 可用于额外的完整 LangSmith Trace；未配置时只保存最终消息轨迹。outputs 中轨迹应包含模型 tool_calls 与对应 tool_call_id 的工具返回。轨迹可能含用户输入，分享前检查。

## 代码对应
- 模型：agent.py 的 ChatOpenAI。
- 业务工具：tools.py 的 analyze_price_volume。
- Harness：agent.py 的 create_deep_agent，执行循环由框架提供。
- Middleware / Backend：目前使用库默认配置，尚未逐一核验，不将其标记为 Basic 完成。

## 打卡清单
- [x] 按用户提供的任务核对 Lite / Basic / Advanced 目标。
- [ ] 核对表单内的额外字段（尚未访问确认）。
- [ ] 第1–2章示例和 notebook 实际运行。
- [ ] 真实模型调用自己的工具，检查请求、返回与最终回答。
- [ ] 整理验证什么、改了什么、实际结果、证据与解释、下一步修正。
- [ ] Basic：启动课程主应用并定位四个组件。
- [ ] Advanced：边界输入与完整 Trace。

## 来源
- 课程：https://datawhalechina.github.io/deepagents-in-action/
- 仓库：https://github.com/datawhalechina/deepagents-in-action
- SDK 快速上手：https://docs.langchain.com/oss/python/deepagents/quickstart

## 下一轮学习
先预测工具输出，再运行核对：价格上涨但近期均量下降时，rise_with_volume_expansion 应是什么？模型应该如何解释？
