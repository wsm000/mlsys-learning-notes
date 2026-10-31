
# DataFlow Task5 构建高质量训练数据 — vm-60 环境
# 复用 Task4 的 AMD Radeon OpenAI 兼容端点（密钥只从远端 key 文件进进程，不落盘不外传）
export PATH="$HOME/.venv-df3/bin:$PATH"
export DF_API_KEY="$(tr -d '\r\n' < /home/simin/codex-task1-20261013/.mimo-key)"
export AMD_API_BASE="https://developer.amd.com.cn/radeon/api/v1"
export DF_MODEL="DeepSeek-V4-Flash"
# vm-60 直连外网被拦截（HTTPS 被重置/劫持到 10.0.31.130）：AMD API 经 Windows 主机 CONNECT 隧道访问
#   vm-60:18899 --(ssh -R)--> Windows:18888(node CONNECT proxy) --> developer.amd.com.cn:443
export HTTPS_PROXY=http://127.0.0.1:18899
export HTTP_PROXY=http://127.0.0.1:18899
export NO_PROXY="127.0.0.1,localhost"
# 模型缓存：离线预置于 /mnt/nfs/task5_hf（gpt2 / PairQual-Scorer-en / deita-quality-scorer / fasttext-lid）
export T5_HF_CACHE="/mnt/nfs/task5_hf"
export HF_HOME="$T5_HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export T5_RPM="${T5_RPM:-10}"
export T5_MAX_WORKERS="${T5_MAX_WORKERS:-2}"
export T5_MAX_RETRIES="${T5_MAX_RETRIES:-6}"
export T5_NUM_SAMPLES="${T5_NUM_SAMPLES:-9}"
export T5_SEED="${T5_SEED:-20261101}"
