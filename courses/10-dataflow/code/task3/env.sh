# DataFlow Task3 environment (vm-60)
export PATH="$HOME/.venv-df3/bin:$PATH"      # so the mineru CLI (called via subprocess by the Local operator) is found
export DF_API_KEY="ollama-local"             # Ollama ignores the key; required by DataFlow serving classes
export MINERU_MODEL_SOURCE=local
export MINERU_TOOLS_CONFIG_JSON="$HOME/mineru.json"
export TIKTOKEN_CACHE_DIR="$HOME/df3_assets/tiktoken_cache"
export PY="$HOME/.venv-df3/bin/python"
