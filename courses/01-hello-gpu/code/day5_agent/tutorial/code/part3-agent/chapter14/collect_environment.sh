#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${KERNEL_AGENT_PYTHON:-$PART_ROOT/.venv/bin/python}"
NODE="${KERNEL_AGENT_NODE:-node}"

uname -a
printf '\nhipcc:\n'
hipcc --version
printf '\nGPU:\n'
rocminfo | awk '/Agent 2/{seen=1} seen && /(Name:|Marketing Name:|Wavefront Size:|Compute Unit:)/{print} seen && /ISA Info:/{exit}'
printf '\nPython packages:\n'
"$PYTHON" -c 'import torch, triton; print(f"python torch={torch.__version__} triton={triton.__version__} device={torch.cuda.get_device_name(0)}")'
printf '\nNode:\n'
"$NODE" --version

