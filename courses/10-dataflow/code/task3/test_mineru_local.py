#!/usr/bin/env python3
"""Day2 smoke test: parse one PDF with local MinerU vlm-transformers backend."""
import os, sys, time, json, glob, subprocess
os.environ.setdefault("MINERU_MODEL_SOURCE", "local")
os.environ.setdefault("MINERU_TOOLS_CONFIG_JSON", os.path.expanduser("~/mineru.json"))

pdf = sys.argv[1] if len(sys.argv) > 1 else "../example_data/PDF2VQAPipeline/questionextract_test.pdf"
out = sys.argv[2] if len(sys.argv) > 2 else "intermediate_smoke"
t0 = time.time()
cmd = ["mineru", "-p", pdf, "-o", out, "-b", "vlm-transformers", "--source", "local"]
print("CMD:", " ".join(cmd), flush=True)
r = subprocess.run(cmd)
print("mineru rc:", r.returncode, "elapsed: %.1fs" % (time.time() - t0))
base = os.path.join(out, os.path.splitext(os.path.basename(pdf))[0], "vlm-transformers")
print("=== output dir:", base)
for f in sorted(glob.glob(base + "/*")):
    print("  ", os.path.basename(f), os.path.getsize(f) if os.path.isfile(f) else "<dir>")
imgs = glob.glob(base + "/images/*")
print("images:", len(imgs), [os.path.basename(i) for i in imgs[:10]])
cl = glob.glob(base + "/*_content_list.json")
if cl:
    data = json.load(open(cl[0], encoding="utf-8"))
    print("content_list items:", len(data))
    from collections import Counter
    print("types:", Counter(d.get("type") for d in data))
    for d in data[:6]:
        print("  sample:", json.dumps(d, ensure_ascii=False)[:150])
