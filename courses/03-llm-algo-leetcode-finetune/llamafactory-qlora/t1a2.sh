#!/usr/bin/env bash
set -x
# 1) 先试无密码 sudo 装 python3-venv
if sudo -n true 2>/dev/null; then
  sudo -n apt-get install -y python3.10-venv >/dev/null 2>&1 && echo SUDO_VENV_OK || echo SUDO_VENV_FAIL
else
  echo NO_PASSWORDLESS_SUDO
fi
# 2) 建 venv（两条路线）
rm -rf ~/venvs/llamafactory
if python3 -m venv --system-site-packages ~/venvs/llamafactory 2>/dev/null; then
  echo VENV_STD_OK
else
  python3 -m venv --without-pip --system-site-packages ~/venvs/llamafactory || exit 1
  echo VENV_NOPIP_OK
  source ~/venvs/llamafactory/bin/activate
  timeout 90 curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py || exit 1
  python /tmp/get-pip.py || exit 1
fi
source ~/venvs/llamafactory/bin/activate
pip --version
# 3) 经 ghfast.top 克隆
cd ~/projects
if [ ! -d LLaMA-Factory ]; then
  git clone --depth 1 https://ghfast.top/https://github.com/hiyouga/LLaMA-Factory.git || exit 1
fi
cd LLaMA-Factory && git log --oneline -1
du -sh . ~/.venvs 2>/dev/null; du -sh ~/venvs/llamafactory
df -h / | tail -1
echo T1A2_DONE
