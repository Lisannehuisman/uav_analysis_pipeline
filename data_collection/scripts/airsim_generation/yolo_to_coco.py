import os
import json
import argparse
from pathlib import Path
from PIL import Image

def load_class_names(classes_path: Path):
    """
    classes.txt format:
      one class name per line
    """
    names = []
    with open(classes_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names

def yolo_line_to_xywh(line, img_w, img_h):
    """
    YOLO: cls xc yc w h (all normalized 0..1)
    COCO bbox: [x_min, y_min, width, height] in pixels
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(parts[0])
    xc, yc, w, h = map(float, parts[1:])

    bw = w * img_w
    bh = h * img_h
    x_min = (xc * img_w) - bw / 2.0
    y_min = (yc * img_h) - bh / 2.0

    # clamp (optional but helps with minor rounding errors)
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    bw = max(0.0, min(bw, img_w - x_min))
    bh = max(0.0, min(bh, img_h - y_min))

    return cls, [x_min, y_min, bw, bh]

def convert_split(
    images_dir: Path,
    labels_dir: Path,
    classes_path: Path,
    out_json: Path,
    img_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
):
    class_names = load_class_names(classes_path)

    coco = {
        "info": {"description": "Converted from YOLO", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }

    image_id = 1
    ann_id = 1

    # collect images
    img_files = []
    for ext in img_exts:
        img_files.extend(images_dir.glob(f"*{ext}"))
        img_files.extend(images_dir.glob(f"*{ext.upper()}"))
    img_files = sorted(set(img_files))

    if not img_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    for img_path in img_files:
        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            }
        )

        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            for ln in lines:
                parsed = yolo_line_to_xywh(ln, w, h)
                if parsed is None:
                    continue
                cls0, bbox = parsed

                # COCO category_id should start at 1
                cat_id = cls0 + 1

                area = float(bbox[2] * bbox[3])
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [float(x) for x in bbox],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1

        image_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"Wrote COCO JSON: {out_json}  (images={len(coco['images'])}, anns={len(coco['annotations'])})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Path to split images folder")
    ap.add_argument("--labels", required=True, help="Path to split labels folder (YOLO txts)")
    ap.add_argument("--classes", required=True, help="Path to classes.txt (one class name per line)")
    ap.add_argument("--out", required=True, help="Output COCO json path")
    args = ap.parse_args()

    convert_split(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        classes_path=Path(args.classes),
        out_json=Path(args.out),
    )


# python yolo_to_coco.py --images /path/to/dataset/train/images --labels /path/to/dataset/train/labels --classes /path/to/dataset/classes.txt --out /path/to/dataset/annotations/instances_train.json