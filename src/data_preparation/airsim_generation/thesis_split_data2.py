import os, random, shutil, re, argparse
from pathlib import Path

# -----------------------
# ARGUMENTS
# -----------------------
parser = argparse.ArgumentParser(
    description="Split AirSim dataset into train/val/test for YOLO"
)
parser.add_argument(
    "capture_name",
    help="Folder name, e.g. S0_20251128_150628"
)
parser.add_argument(
    "--root",
    default=str(Path(__file__).resolve().parents[2] / "raw_data"),
    help="Base captures directory"
)
parser.add_argument(
    "--split",
    choices=["random", "by_angle", "by_object"],
    default="random",
    help="Split strategy"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)

args = parser.parse_args()

# -----------------------
# PATHS
# -----------------------
ROOT = os.path.join(args.root, args.capture_name)
OUT  = os.path.join(ROOT, "dataset")
SPLIT_MODE = args.split

random.seed(args.seed)

# -----------------------
# CREATE FOLDERS
# -----------------------
for sub in [
    "images/train","images/val","images/test",
    "labels/train","labels/val","labels/test"
]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

# -----------------------
# LOAD FILES
# -----------------------
img_dir = os.path.join(ROOT, "images")
lbl_dir = os.path.join(ROOT, "labels")

imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]

# -----------------------
# HELPERS
# -----------------------
def angle_bucket(fname):
    m = re.search(r"-az(\d{3})-", fname)
    return None if not m else int(m.group(1))

def object_key(fname):
    m = re.match(r"[^-]+-([^-]+)-L", fname)
    return m.group(1) if m else "UNK"

# -----------------------
# GROUPINGS
# -----------------------
by_angle = {}
by_obj   = {}

for f in imgs:
    by_angle.setdefault(angle_bucket(f), []).append(f)
    by_obj.setdefault(object_key(f), []).append(f)

train, val, test = [], [], []

# -----------------------
# SPLITS
# -----------------------
if SPLIT_MODE == "random":
    random.shuffle(imgs)
    n = len(imgs)
    n_train = round(0.70 * n)
    n_val   = round(0.15 * n)

    train = imgs[:n_train]
    val   = imgs[n_train:n_train+n_val]
    test  = imgs[n_train+n_val:]

elif SPLIT_MODE == "by_angle":
    buckets = [k for k in by_angle if k is not None]
    random.shuffle(buckets)

    test_b = buckets[0]
    val_b  = buckets[1] if len(buckets) > 1 else buckets[0]

    test  = by_angle[test_b]
    val   = by_angle[val_b]
    train = [
        f for k, fs in by_angle.items()
        if k not in (test_b, val_b)
        for f in fs
    ]

elif SPLIT_MODE == "by_object":
    objs = list(by_obj.keys())
    random.shuffle(objs)

    test_o = objs[0]
    val_o  = objs[1] if len(objs) > 1 else objs[0]

    test  = by_obj[test_o]
    val   = by_obj[val_o]
    train = [
        f for k, fs in by_obj.items()
        if k not in (test_o, val_o)
        for f in fs
    ]

else:
    raise ValueError("Unknown split mode")

# -----------------------
# COPY FILES
# -----------------------
def move_pair(f, split):
    src_img = os.path.join(img_dir, f)
    src_lbl = os.path.join(lbl_dir, f.replace(".png", ".txt"))

    dst_img = os.path.join(OUT, "images", split, f)
    dst_lbl = os.path.join(OUT, "labels", split, f.replace(".png", ".txt"))

    # 🔑 ENSURE DESTINATION FOLDERS EXIST
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)

    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)


for f in train: move_pair(f, "train")
for f in val:   move_pair(f, "val")
for f in test:  move_pair(f, "test")

print(
    f"[{args.capture_name}] "
    f"Train {len(train)}, Val {len(val)}, Test {len(test)} "
    f"→ {OUT}"
)
