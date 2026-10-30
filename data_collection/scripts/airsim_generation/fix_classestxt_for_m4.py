import json
from pathlib import Path

COCO_DIR = Path(__file__).resolve().parents[2] / "raw_data" / "synthetic_subset" / "coco_annotations"

m3_path = COCO_DIR / "coco_instances_val_M3.json"
m4_path = COCO_DIR / "coco_instances_val_M4.json"
out_path = COCO_DIR / "coco_instances_val_M4_fixed.json"

def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save(d, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)

print("Loading M3 (reference):", m3_path)
print("Loading M4 (to fix):", m4_path)

m3 = load(m3_path)
m4 = load(m4_path)

# ---- REPLACE CATEGORIES ----
m4["categories"] = sorted(m3["categories"], key=lambda c: c["id"])

# ---- SANITY CHECK ----
cat_ids = set(c["id"] for c in m4["categories"])
ann_ids = set(a["category_id"] for a in m4["annotations"])
missing = sorted(list(ann_ids - cat_ids))

print("Categories:", len(cat_ids), "max id:", max(cat_ids))
print("Annotation category ids:", sorted(ann_ids))
print("Missing ids:", missing)

if missing:
    raise RuntimeError(f"❌ Still missing category ids: {missing}")

save(m4, out_path)

print("\n✅ FIX SUCCESSFUL")
print("Wrote:", out_path)
