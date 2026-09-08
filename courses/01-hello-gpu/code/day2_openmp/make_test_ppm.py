from pathlib import Path
import sys

w = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
h = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/input.ppm"

pixels = bytearray()
for y in range(h):
    for x in range(w):
        # Deterministic colorful pattern with enough work for benchmarking.
        pixels.extend(((x * 255 // max(1, w - 1)) & 255,
                       (y * 255 // max(1, h - 1)) & 255,
                       ((x // 32 + y // 32) * 37) & 255))
Path(path).write_bytes(f"P6\n{w} {h}\n255\n".encode() + pixels)
print(f"wrote {path}: {w}x{h}")
