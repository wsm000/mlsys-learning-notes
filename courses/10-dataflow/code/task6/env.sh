
# DataFlow Task6 模型性能评测 — vm-60 环境
# 复用 Task4/5 的 AMD Radeon OpenAI 兼容端点（密钥只从远端 key 文件进进程，不落盘不外传）
export PATH="$HOME/.venv-df3/bin:$PATH"
export DF_API_KEY="$(tr -d '\r\n' < /home/simin/codex-task1-20261013/.mimo-key)"
export AMD_API_BASE="https://developer.amd.com.cn/radeon/api/v1"
export DF_MODEL="DeepSeek-V4-Flash"
# 2026-11-02 实测：vm-60 到 developer.amd.com.cn 直连 200（Task5 时期的劫持/隧道已不需要），
# 但端点仍是 20 requests/min 用户级限流，客户端仍需匀速放行。
export T6_RPM="${T6_RPM:-10}"
export T6_MAX_WORKERS="${T6_MAX_WORKERS:-2}"
export T6_MAX_RETRIES="${T6_MAX_RETRIES:-6}"
# 本地被测模型（替代官方 Qwen/Qwen2.5-7B-Instruct；24GB 卡放不下 7B+vllm 双副本，且 vllm 与现有 torch 冲突）
# 已从 /mnt/nfs（NFS 93% 满，加载一次 91s）复制到本地盘，加载 ~10s
export T6_MODEL_PATH="${T6_MODEL_PATH:-$HOME/dataflow-task6/model/Qwen2.5-0.5B-Instruct}"
export T6_TEMPERATURE="${T6_TEMPERATURE:-0}"
export T6_MAX_TOKENS="${T6_MAX_TOKENS:-512}"
export T6_MAX_MODEL_LEN="${T6_MAX_MODEL_LEN:-2048}"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
