#!/usr/bin/env python3
"""
build_stereo_dataset.py — собрать тренируемый YOLO-OBB датасет из двух бэкапов:
  experiments/backups/stereo_A_242labels_yolo_obb_2026-04-22.zip   (YOLO-OBB, LEFT 0000-0241)
  experiments/backups/stereo_B_258labels_cvat_xml_2026-04-22.zip   (CVAT XML, RIGHT 0400-0658)

Картинки берутся из experiments/dataset_stereo_v1/images/ (train+val pool).
На выходе — experiments/dataset_stereo_final/ c 80/20 split, только размеченные кадры.
Дополнительно — visual check в experiments/dataset_stereo_final/labels_check/.
"""
import math
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUPS = ROOT / "backups"
ZIP_A = BACKUPS / "stereo_A_242labels_yolo_obb_2026-04-22.zip"
ZIP_B = BACKUPS / "stereo_B_258labels_cvat_xml_2026-04-22.zip"

# Источник картинок — каталог images/ исходного dataset_stereo_v1.
# Задаётся переменной окружения STEREO_V1_IMAGES, по умолчанию — рядом с репозиторием.
_ARCHIVE = Path(os.environ.get("STEREO_V1_IMAGES", ROOT / "dataset_stereo_v1" / "images"))
IMG_POOL_DIRS = [_ARCHIVE / "train", _ARCHIVE / "val"]

OUT_DIR = ROOT / "dataset_stereo_final"
TMP_DIR = Path("/tmp/_stereo_build")

# Split: 70% train, 15% val, 15% test (for YOLO train/val split monitoring
# during training + test split for final unseen-data eval on defense).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# TEST_FRACTION is 1 - TRAIN_FRACTION - VAL_FRACTION = 0.15
SEED = 42
W = 640
H = 480

# Convert to grayscale (3-channel duplicated) to match inference pipeline
# (yolo_detector.py converts RGB→gray→3ch before inference).
CONVERT_TO_GRAYSCALE = True


def unpack_zips():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    (TMP_DIR / "A").mkdir()
    (TMP_DIR / "B").mkdir()
    with zipfile.ZipFile(ZIP_A) as z:
        z.extractall(TMP_DIR / "A")
    with zipfile.ZipFile(ZIP_B) as z:
        z.extractall(TMP_DIR / "B")


def collect_image_pool():
    """Returns dict: stem → absolute path of source image."""
    pool = {}
    for d in IMG_POOL_DIRS:
        for p in d.glob("scene_*.png"):
            pool[p.stem] = p
    return pool


# ── A: YOLO-OBB labels are already normalized, just copy as-is ────────────────

def read_a_labels():
    """Returns dict: stem → label text (YOLO-OBB, already normalized)."""
    labels_dir = TMP_DIR / "A" / "labels" / "train"
    out = {}
    for p in sorted(labels_dir.glob("*.txt")):
        text = p.read_text().strip()
        if text:
            out[p.stem] = text
    return out


# ── B: CVAT XML box (xtl, ytl, xbr, ybr, rotation deg) → YOLO-OBB 4 corners ──

def rotate_point(px, py, cx, cy, theta_rad):
    """Rotate (px, py) around (cx, cy). Image coords (y-axis down)."""
    dx, dy = px - cx, py - cy
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return (cx + dx * c - dy * s, cy + dx * s + dy * c)


def cvat_box_to_obb_corners(xtl, ytl, xbr, ybr, rotation_deg):
    """Convert CVAT rotated-box to 4 corners in pixel coords (TL, TR, BR, BL)."""
    cx = (xtl + xbr) / 2.0
    cy = (ytl + ybr) / 2.0
    theta = math.radians(rotation_deg)
    # Original axis-aligned corners
    corners = [
        (xtl, ytl),  # TL
        (xbr, ytl),  # TR
        (xbr, ybr),  # BR
        (xtl, ybr),  # BL
    ]
    return [rotate_point(px, py, cx, cy, theta) for px, py in corners]


def read_b_labels():
    """Returns dict: stem → label text (YOLO-OBB, normalized).
    Only entries that have ≥1 box."""
    xml_path = TMP_DIR / "B" / "annotations.xml"
    root = ET.parse(xml_path).getroot()
    out = {}
    for img in root.findall("image"):
        name = img.get("name")       # "scene_0401_right.png"
        iw = int(img.get("width"))   # 640
        ih = int(img.get("height"))  # 480
        stem = os.path.splitext(name)[0]
        lines = []
        for box in img.findall("box"):
            if box.get("label") != "box":
                continue
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            rot = float(box.get("rotation") or 0.0)
            corners = cvat_box_to_obb_corners(xtl, ytl, xbr, ybr, rot)
            # Normalize to [0, 1], clip to valid range
            norm = []
            for x, y in corners:
                nx = max(0.0, min(1.0, x / iw))
                ny = max(0.0, min(1.0, y / ih))
                norm.extend([nx, ny])
            lines.append("0 " + " ".join(f"{v:.6f}" for v in norm))
        if lines:
            out[stem] = "\n".join(lines)
    return out


def make_splits(stems, train_fraction, val_fraction, seed):
    """Split stems into train/val/test sets. Remaining = test."""
    rng = random.Random(seed)
    stems = sorted(stems)                 # deterministic base ordering
    rng.shuffle(stems)
    n_total = len(stems)
    n_train = int(round(n_total * train_fraction))
    n_val = int(round(n_total * val_fraction))
    train = set(stems[:n_train])
    val = set(stems[n_train:n_train + n_val])
    test = set(stems[n_train + n_val:])
    return train, val, test


def save_image(src_path, dst_path, to_grayscale):
    """Copy image, optionally converting to 3-channel grayscale."""
    if not to_grayscale:
        shutil.copy2(src_path, dst_path)
        return
    import cv2
    img = cv2.imread(str(src_path))  # BGR
    # cv2's COLOR_BGR2GRAY uses same ITU-R BT.601 weights (0.299R + 0.587G + 0.114B)
    # as yolo_detector.py — inference and training see identical pixels.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(dst_path), bgr3)


def write_dataset(labels_by_stem, img_pool):
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (OUT_DIR / sub).mkdir(parents=True)

    stems = sorted(labels_by_stem.keys())
    missing_imgs = [s for s in stems if s not in img_pool]
    if missing_imgs:
        print(f"[warn] {len(missing_imgs)} labels без изображения (пропущены):")
        for s in missing_imgs[:5]:
            print(f"       {s}")
    stems = [s for s in stems if s in img_pool]

    train_set, val_set, test_set = make_splits(stems, TRAIN_FRACTION, VAL_FRACTION, SEED)

    counts = {k: 0 for k in ("train", "val", "test",
                             "train_left", "train_right",
                             "val_left", "val_right",
                             "test_left", "test_right", "boxes")}
    for s in stems:
        if s in train_set:
            split = "train"
        elif s in val_set:
            split = "val"
        else:
            split = "test"
        src = img_pool[s]
        dst_img = OUT_DIR / "images" / split / f"{s}.png"
        dst_lbl = OUT_DIR / "labels" / split / f"{s}.txt"
        save_image(src, dst_img, CONVERT_TO_GRAYSCALE)
        dst_lbl.write_text(labels_by_stem[s] + "\n")
        counts[split] += 1
        cam = "left" if "_left" in s else "right"
        counts[f"{split}_{cam}"] += 1
        counts["boxes"] += labels_by_stem[s].count("\n") + 1

    yaml = (
        f"path: {OUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['box']\n"
    )
    (OUT_DIR / "dataset.yaml").write_text(yaml)
    return counts


def draw_labels_check(labels_by_stem, img_pool, n=12):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[labels_check] opencv-python / numpy не установлены на хосте — пропускаю визуальную проверку.")
        print("               pip3 install opencv-python numpy --break-system-packages")
        return

    check_dir = OUT_DIR / "labels_check"
    check_dir.mkdir(exist_ok=True)
    rng = random.Random(SEED + 1)
    stems = [s for s in labels_by_stem if s in img_pool]
    sample = rng.sample(stems, min(n, len(stems)))
    for s in sample:
        img = cv2.imread(str(img_pool[s]))
        h, w = img.shape[:2]
        for line in labels_by_stem[s].splitlines():
            parts = line.split()
            coords = list(map(float, parts[1:]))
            pts = np.array([[coords[i] * w, coords[i + 1] * h] for i in range(0, 8, 2)], dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.imwrite(str(check_dir / f"{s}.png"), img)
    print(f"[labels_check] {len(sample)} примеров в {check_dir}")


def main():
    print("[1/4] Распаковка бэкапов…")
    unpack_zips()

    print("[2/4] Чтение разметок…")
    labels_a = read_a_labels()
    labels_b = read_b_labels()
    print(f"       A (LEFT,  YOLO-OBB): {len(labels_a)} кадров")
    print(f"       B (RIGHT, CVAT XML): {len(labels_b)} кадров с ≥1 box")

    labels_all = {**labels_a, **labels_b}
    assert not (set(labels_a) & set(labels_b)), "overlap между A и B (левые/правые кадры?)"
    print(f"       total: {len(labels_all)} размеченных кадров")

    print("[3/4] Поиск картинок в dataset_stereo_v1/images/…")
    img_pool = collect_image_pool()
    print(f"       картинок в пуле: {len(img_pool)}")

    mode = "grayscale-3ch" if CONVERT_TO_GRAYSCALE else "RGB"
    print(f"[4/4] Сборка датасета в {OUT_DIR} ({mode})…")
    counts = write_dataset(labels_all, img_pool)
    print(f"       train: {counts['train']} (left={counts['train_left']}, right={counts['train_right']})")
    print(f"       val:   {counts['val']} (left={counts['val_left']}, right={counts['val_right']})")
    print(f"       test:  {counts['test']} (left={counts['test_left']}, right={counts['test_right']})")
    print(f"       всего box-объектов: {counts['boxes']}")

    # Draw labels check from the OUTPUT dir (grayscale), so user sees what the model sees.
    out_img_pool = {p.stem: p for p in (OUT_DIR / "images").rglob("*.png")}
    draw_labels_check(labels_all, out_img_pool, n=12)


if __name__ == "__main__":
    main()
