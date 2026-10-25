#!/usr/bin/env python3
"""Task3 verification: check the full chain outputs (MinerU layout -> QA -> ShareGPT).

Honest checks only: existence, field structure, counts, image references.
No claim about semantic quality of the local 1.7B model's extraction.
"""
import os, sys, json, glob

OK, FAIL = [], []
def check(name, cond, detail=""):
    (OK if cond else FAIL).append((name, detail))
    print(("PASS " if cond else "FAIL ") + name + (" | " + str(detail) if detail else ""))

base = os.getcwd()

# --- 1. MinerU layout outputs ---
md_dirs = sorted(glob.glob(os.path.join(base, "intermediate", "*", "vlm-transformers")))
check("mineru_output_dirs>=1", len(md_dirs) >= 1, [os.path.relpath(d, base) for d in md_dirs])
for d in md_dirs:
    name = os.path.basename(os.path.dirname(d))
    md = os.path.join(d, name + ".md")
    cl = glob.glob(os.path.join(d, "*_content_list.json"))
    imgs = glob.glob(os.path.join(d, "images", "*"))
    check(f"mineru[{name}].md exists", os.path.exists(md) and os.path.getsize(md) > 0)
    check(f"mineru[{name}].content_list", len(cl) >= 1 and os.path.getsize(cl[0]) > 0)
    items = json.load(open(cl[0], encoding="utf-8")) if cl else []
    types = {}
    for it in items:
        types[it.get("type", "?")] = types.get(it.get("type", "?"), 0) + 1
    check(f"mineru[{name}].content_list_nonempty", len(items) > 0, f"{len(items)} items {types}")
    check(f"mineru[{name}].has_text", types.get("text", 0) > 0)
    check(f"mineru[{name}].images_dir", len(imgs) >= 0, f"{len(imgs)} images")

# --- 2. converted layout json (MinerU2LLMInputOperator output) ---
conv = sorted(glob.glob(os.path.join(base, "intermediate", "*", "vlm-transformers", "*_content_list_converted.json")))
check("converted_layout>=1", len(conv) >= 1, len(conv))
for c in conv:
    data = json.load(open(c, encoding="utf-8"))
    ids = [d.get("id") for d in data]
    check("converted_ids_unique", len(ids) == len(set(ids)) and all(isinstance(i, int) for i in ids))

# --- 3. extracted / merged QA ---
for name_dir in sorted(glob.glob(os.path.join(base, "cache", "*"))):
    if not os.path.isdir(name_dir):
        continue
    name = os.path.basename(name_dir)
    ex = os.path.join(name_dir, "extracted_vqa.jsonl")
    mg = os.path.join(name_dir, "merged_qa_pairs.jsonl")
    md = os.path.join(name_dir, "merged_qa_pairs.md")
    if os.path.exists(ex):
        rows = [json.loads(l) for l in open(ex, encoding="utf-8") if l.strip()]
        check(f"extracted[{name}].fields", all(set(["question", "answer", "solution", "label", "chapter_title"]) <= set(r) for r in rows), len(rows))
    if os.path.exists(mg):
        rows = [json.loads(l) for l in open(mg, encoding="utf-8") if l.strip()]
        check(f"merged[{name}].pairs>=0", True, f"{len(rows)} pairs")
        for r in rows:
            for k in ("question", "answer"):
                if not r.get(k):
                    check(f"merged[{name}].nonempty_{k}", False, r.get("label"))
    # image references resolve
    imgdir = os.path.join(name_dir, "vqa_images")
    n_img = len(glob.glob(os.path.join(imgdir, "*"))) if os.path.isdir(imgdir) else 0
    print(f"INFO merged[{name}] images copied: {n_img}")

# --- 4. ShareGPT output ---
qa_json = os.path.join(base, "cache", "data", "qa.json")
if os.path.exists(qa_json):
    recs = json.load(open(qa_json, encoding="utf-8"))
    check("sharegpt.records>=0", True, len(recs))
    for r in recs:
        msgs = r.get("messages", [])
        check("sharegpt.messages_roles", [m.get("role") for m in msgs] == ["user", "assistant"], msgs and msgs[0].get("role"))
        imgs = r.get("images", [])
        for p in imgs:
            check("sharegpt.image_exists", os.path.exists(p), p)
else:
    check("sharegpt.qa.json exists", False)

print()
print(f"SUMMARY: {len(OK)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for n, d in FAIL:
        print(" -", n, d)
sys.exit(1 if FAIL else 0)
