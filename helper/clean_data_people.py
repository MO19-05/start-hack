import os
import shutil
from PIL import Image

# === CONFIGURATION ===
DATA_ROOT = "C:\\Users\\moham\\Downloads\\jhu_crowd_v2.0\\filtered_dataset"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
ANNOT_DIR = os.path.join(DATA_ROOT, "gt")
LABEL_FILE = os.path.join(DATA_ROOT, "image_labels.txt")




OUTPUT_DATASET = "C:\\Users\\moham\\Downloads\\jhu_crowd_v2.0\\yolo_dataset"
IMG_OUT = os.path.join(OUTPUT_DATASET, "images")
LBL_OUT = os.path.join(OUTPUT_DATASET, "labels")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

def convert_to_yolo(x, y, w, h, img_w, img_h):
    # convert top-left → YOLO center format
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

# Build a YOLOv11-style dataset with normalized center-format labels
# Create split folders and write `data.yaml`.
VALID_EXTS = (".jpg", ".jpeg", ".png")
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test
RNG_SEED = 42


def make_dirs(root):
    # create flat images/ and labels/ folders (no train/val/test split)
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)


def write_data_yaml(root):
    yaml_content = (
        "path: ./\n"
        # single-folder dataset (all images in images/ and labels in labels/)
        f"train: {os.path.join('images').replace('\\', '/')}\n"
        "nc: 1\n"
        "names: ['person']\n"
    )
    with open(os.path.join(OUTPUT_DATASET, "data.yaml"), "w", encoding="utf-8") as yf:
        yf.write(yaml_content)


def gather_images_with_ann(image_dir, annot_dir):
    pairs = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(VALID_EXTS):
            continue
        image_id = os.path.splitext(fname)[0]
        ann_path = os.path.join(annot_dir, image_id + ".txt")
        img_path = os.path.join(image_dir, fname)
        if os.path.exists(ann_path):
            pairs.append((img_path, ann_path, fname))
    return pairs


def split_pairs(pairs, ratios=SPLIT_RATIOS, seed=RNG_SEED):
    import random

    random.seed(seed)
    pairs = pairs.copy()
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # ensure counts add up
    n_test = n - n_train - n_val
    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def process_and_write(splits):
    # For flat dataset: write all labels to LBL_OUT and copy all images to IMG_OUT
    os.makedirs(LBL_OUT, exist_ok=True)
    os.makedirs(IMG_OUT, exist_ok=True)
    for split_name, items in splits.items():
        for img_path, ann_path, fname in items:
            # read image size (kept for potential future use)
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                img_w = img_h = None

            yolo_lines = []
            with open(ann_path, "r", encoding="utf-8") as af:
                for line in af:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    try:
                        x, y, w, h = map(int, parts[:4])
                    except ValueError:
                        continue
                    x_int = int(round(x))
                    y_int = int(round(y))
                    w_int = int(round(w))
                    h_int = int(round(h))
                    yolo_lines.append(f"{x_int} {y_int} {x_int + w_int} {y_int + h_int} 0")

            base = os.path.splitext(fname)[0]
            out_label_path = os.path.join(LBL_OUT, base + ".txt")
            with open(out_label_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(yolo_lines))

            # copy image into flat images directory
            shutil.copy2(img_path, os.path.join(IMG_OUT, fname))


def build_yolov11_dataset():
    # Create flat dataset structure and process all annotated images into it
    make_dirs(OUTPUT_DATASET)
    pairs = gather_images_with_ann(IMAGE_DIR, ANNOT_DIR)
    if not pairs:
        print("No annotated images found — nothing to do.")
        return
    # create a single pseudo-split dictionary to reuse the processing function
    splits = {"all": pairs}
    process_and_write(splits)
    write_data_yaml(OUTPUT_DATASET)
    print("✓ YOLOv11 flat dataset created at:", OUTPUT_DATASET)


if __name__ == "__main__":
    build_yolov11_dataset()
print("✓ Conversion complete!")
