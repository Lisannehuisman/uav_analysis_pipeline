import os
import shutil
import re
from pathlib import Path

# =========================
# CONFIG
# =========================
ROOT = str(Path(__file__).resolve().parents[2] / "raw_data" / "synthetic_subset")

IMG_TRAIN = os.path.join(ROOT, "images", "train")
LBL_TRAIN = os.path.join(ROOT, "labels", "train")

# Kies hier je "canonical" instellingen
AZ_CANON   = 0          # bv 0 graden
ELEV_CANON = "mid"      # low/mid/high
RAD_FIXED  = "mid"      # near/mid/far

# Optie 1 voor M2a: 1 azimuth (zelfde als canonical of bewust anders)
M2A_AZ = AZ_CANON

# Optie 2 voor M2b: beperkte set azimuths (robuster)
M2B_AZ_SET = {0, 90, 180, 270}

ELEV_ALL = {"low", "mid", "high"}  # jouw elevatie bins


# =========================
# HELPERS
# =========================
def parse_view(fname: str):
    """
    Matches filenames like:
      S0-SM_barrel_1-elhigh-radfar-az000.png
    """
    el_m  = re.search(r"-el(low|mid|high)-", fname)
    rad_m = re.search(r"-rad(near|mid|far)-", fname)
    az_m  = re.search(r"-az(\d{3})\.", fname)  # az000, az045, ...

    if not (az_m and el_m and rad_m):
        raise ValueError(f"Cannot parse view from filename: {fname}")

    az = int(az_m.group(1))      # 000 -> 0, 045 -> 45
    el = el_m.group(1)           # low/mid/high
    rad = rad_m.group(1)         # near/mid/far
    return az, el, rad

def copy_pair(img_src, lbl_src, img_dst_dir, lbl_dst_dir):
    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)
    shutil.copy(img_src, os.path.join(img_dst_dir, os.path.basename(img_src)))
    shutil.copy(lbl_src, os.path.join(lbl_dst_dir, os.path.basename(lbl_src)))


# =========================
# FILTERS PER MODEL
# =========================
def keep_M1(az, el, rad):
    # Canonical: 1 az × 1 elev × 1 rad
    return (az == AZ_CANON) and (el == ELEV_CANON) and (rad == RAD_FIXED)

def keep_M2a(az, el, rad):
    # Optie 1: 1 az × alle elevaties × 1 rad
    return (az == M2A_AZ) and (el in ELEV_ALL) and (rad == RAD_FIXED)

def keep_M2b(az, el, rad):
    # Optie 2: beperkte az-set × alle elevaties × 1 rad
    return (az in M2B_AZ_SET) and (el in ELEV_ALL) and (rad == RAD_FIXED)

def keep_M3(az, el, rad):
    # Alle az × 1 elev × 1 rad
    return (el == ELEV_CANON) and (rad == RAD_FIXED)

def keep_M4(az, el, rad):
    # Full multiview
    return True


FILTERS = {
    "M1":  keep_M1,
    "M2a": keep_M2a,   # Optie 1
    "M2b": keep_M2b,   # Optie 2
    "M3":  keep_M3,
    "M4":  keep_M4,
}


# =========================
# BUILD TRAIN SUBSETS
# =========================
def build_subset(model_name, rule_fn):
    img_out = os.path.join(ROOT, "images", f"train_{model_name}")
    lbl_out = os.path.join(ROOT, "labels", f"train_{model_name}")

    kept = 0
    skipped = 0

    for fname in os.listdir(IMG_TRAIN):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        try:
            az, el, rad = parse_view(fname)
        except ValueError:
            skipped += 1
            continue

        if rule_fn(az, el, rad):
            img_src = os.path.join(IMG_TRAIN, fname)
            lbl_src = os.path.join(LBL_TRAIN, os.path.splitext(fname)[0] + ".txt")

            if not os.path.exists(lbl_src):
                # Als je soms images zonder labels hebt (bijv. leeg), kun je dit aanpassen
                skipped += 1
                continue

            copy_pair(img_src, lbl_src, img_out, lbl_out)
            kept += 1
        else:
            skipped += 1

    print(f"[{model_name}] kept={kept}, skipped={skipped}")
    return kept


if __name__ == "__main__":
    print("IMG_TRAIN =", IMG_TRAIN)
    print("Exists?  ", os.path.exists(IMG_TRAIN))
    files = os.listdir(IMG_TRAIN)
    print("#files   =", len(files))
    print("First 10 filenames:")
    for f in files[:10]:
        print(" ", repr(f))

    print("\nParse test on first 10:")
    for f in files[:10]:
        try:
            print(" ", f, "->", parse_view(f))
        except Exception as e:
            print(" FAILED:", repr(f), "ERR:", e)

    print("\nBuilding derived train splits...")
    counts = {}
    for name, fn in FILTERS.items():
        counts[name] = build_subset(name, fn)

    print("\nDone. Train images per model:")
    for k, v in counts.items():
        print(f"  {k}: {v}")



    print("Building derived train splits...")
    counts = {}
    for name, fn in FILTERS.items():
        counts[name] = build_subset(name, fn)

    print("\nDone. Train images per model:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
 
