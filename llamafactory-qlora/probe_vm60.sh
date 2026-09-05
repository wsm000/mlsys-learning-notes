#!/usr/bin/env bash
echo ===DISK===; df -h / | tail -1
echo ===CACHE===; du -sh ~/.cache/huggingface 2>/dev/null; du -sh ~/.cache/pip 2>/dev/null; du -sh ~/.cache 2>/dev/null
echo ===HF_HUB===; ls ~/.cache/huggingface/hub 2>/dev/null | head -15
echo ===PKGS===; python3 -m pip list 2>/dev/null | grep -iE "llamafactory|swanlab|trl|deepspeed|flash|vllm|sentencepiece|tiktoken"
echo ===RES===; free -g | sed -n 2p; nproc
echo ===HOME_TOP===; du -sh ~/* 2>/dev/null | sort -rh | head -8
echo ===LLF_FIND===; find $HOME -maxdepth 3 -iname "*llama*factory*" 2>/dev/null | head -5
echo ===UV===; which uv || true
echo ===HF_ENV===; grep -i "HF_ENDPOINT" ~/.bashrc 2>/dev/null | head -3 || true
echo ===NET_MIRROR===; timeout 8 curl -sI https://hf-mirror.com 2>&1 | head -2 || true
echo ===NET_HF===; timeout 8 curl -sI https://huggingface.co 2>&1 | head -2 || true
echo ===PROBE_DONE===
