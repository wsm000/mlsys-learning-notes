"""Reuse course 8 credential in memory; never print or copy it."""
import os
from pathlib import Path
from agent import main

if __name__ == "__main__":
    os.environ["AMD_API_KEY"] = Path("/home/simin/codex-task1-20261013/.mimo-key").read_text().strip()
    main()
