import os
import signal
from pathlib import Path
pid = int(Path('/home/simin/course09-quanty/outputs/main-gateway.pid').read_text())
cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
if b'bub' not in cmd or b'gateway' not in cmd:
    raise SystemExit('PID does not identify our gateway; refusing to stop')
os.kill(pid, signal.SIGTERM)
