from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a CVAT XML image-annotation export to YOLO txt labels.")
    parser.add_argument("--xml", required=True, type=Path, help="Path to the CVAT XML file.")
    parser.add_argument("--data-yaml", required=True, type=Path, help="YOLO data YAML with class id mapping.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where YOLO txt labels will be written.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing txt files in the output directory.",
    )
    return parser.parse_args()


def parse_names_from_data_yaml(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    class_name_to_id: dict[str, int] = {}
    in_names = False
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if not raw_line.startswith("  "):
            break
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key_text, value_text = stripped.split(":", 1)
        class_id = int(key_text.strip())
        class_name = value_text.strip().strip("'\"")
        class_name_to_id[class_name] = class_id
    if not class_name_to_id:
        raise ValueError(f"Could not parse class names from '{path}'.")
    return class_name_to_id


def to_yolo_line(class_id: int, xtl: float, ytl: float, xbr: float, ybr: float, width: int, height: int) -> str:
    x_center = ((xtl + xbr) / 2.0) / width
    y_center = ((ytl + ybr) / 2.0) / height
    box_width = (xbr - xtl) / width
    box_height = (ybr - ytl) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def main() -> None:
    args = parse_args()
    xml_path = args.xml.resolve()
    data_yaml = args.data_yaml.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_name_to_id = parse_names_from_data_yaml(data_yaml)
    root = ET.parse(xml_path).getroot()

    image_count = 0
    box_count = 0

    for image_node in root.findall("image"):
        image_count += 1
        image_name = image_node.attrib["name"]
        width = int(image_node.attrib["width"])
        height = int(image_node.attrib["height"])
        label_path = output_dir / f"{Path(image_name).stem}.txt"

        if label_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file '{label_path}'. Re-run with --overwrite.")

        lines: list[str] = []
        for box in image_node.findall("box"):
            class_name = box.attrib["label"]
            if class_name not in class_name_to_id:
                raise KeyError(f"Label '{class_name}' is not present in '{data_yaml}'.")
            class_id = class_name_to_id[class_name]
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            lines.append(to_yolo_line(class_id, xtl, ytl, xbr, ybr, width, height))
            box_count += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"Wrote YOLO labels for {image_count} images to '{output_dir}'.")
    print(f"Converted {box_count} boxes from '{xml_path}'.")


if __name__ == "__main__":
    main()
