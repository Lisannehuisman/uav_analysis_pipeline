import os, json, argparse

def slugify(k):  # nice YOLO names from prefixes
    return k.replace("SM_","").lower()

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", required=True)
ap.add_argument("--classmap", required=True)
ap.add_argument("--out", default=None)
args = ap.parse_args()

with open(args.classmap,"r") as f:
    classmap = json.load(f)

names = {int(v): slugify(k) for k,v in classmap.items()}

# FIX: clean path *before* using in f-string
clean_path = args.dataset_root.replace("\\", "/")

yaml = f"""path: {clean_path}
train: images/train
val: images/val
test: images/test
names:
"""
for k in sorted(names.keys()):
    yaml += f"  {k}: {names[k]}\n"

outp = args.out or os.path.join(args.dataset_root, "data.yaml")

with open(outp, "w") as f:
    f.write(yaml)

print("Wrote:", outp)
