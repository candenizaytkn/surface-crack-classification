"""
Aşama 2: Veri hazırlığı pipeline
D:\data\ altına yazar. Tune/lock setine dokunmaz.
"""

import math, random, shutil
import cv2, numpy as np
from pathlib import Path
from collections import defaultdict

# ── Yollar ──────────────────────────────────────────────────────────────────
POSITIVE_DIR = Path(r"D:\Positive")
NEGATIVE_DIR = Path(r"D:\Negative")
TUNE_DIR     = Path(r"C:\Users\cande\OneDrive\Masaüstü\test images\tune")
LOCK_DIR     = Path(r"C:\Users\cande\OneDrive\Masaüstü\test images\lock")
DATA_DIR     = Path(r"D:\data")

CLASSES    = ["longitudinal", "transverse", "cross", "no_crack"]
RAW_DIRS   = {c: DATA_DIR / "raw"   / c for c in CLASSES}
TRAIN_DIRS = {c: DATA_DIR / "train" / c for c in CLASSES}
VAL_DIRS   = {c: DATA_DIR / "val"   / c for c in CLASSES}

IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIN_LINES   = 3
AUG_TARGET  = 8000
SEED        = 42


def make_dirs():
    for d in list(RAW_DIRS.values()) + list(TRAIN_DIRS.values()) + list(VAL_DIRS.values()):
        d.mkdir(parents=True, exist_ok=True)


# ── V1 sınıflandırıcı ───────────────────────────────────────────────────────
def classify_v1(img_path: Path) -> str:
    buf = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "uncertain"
    blur  = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=30, minLineLength=20, maxLineGap=10)
    if lines is None or len(lines) < MIN_LINES:
        return "uncertain"
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = 90.0 if x2 == x1 else abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if a > 90:
            a = 180 - a
        angles.append(a)
    mean_a = float(np.mean(angles))
    std_a  = float(np.std(angles))
    if std_a  > 25:  return "cross"
    if mean_a >= 60: return "longitudinal"
    if mean_a <= 30: return "transverse"
    return "cross"


# ── Augmentation ─────────────────────────────────────────────────────────────
def augment_img(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    result = img.copy()

    # horizontal/vertical flip
    if rng.random() < 0.5:
        result = cv2.flip(result, rng.choice([0, 1]))

    # rotation ±10°
    angle = rng.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # zoom ±0.1 (affine scale, output boyutu korunur)
    scale = 1.0 + rng.uniform(-0.1, 0.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # contrast ±0.2
    alpha = 1.0 + rng.uniform(-0.2, 0.2)
    result = np.clip(result.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    return result


# ══════════════════════════════════════════════════════════════════════════════
def main():
    make_dirs()

    # ── ADIM 1: Manuel set dosya adları ─────────────────────────────────────
    print("=" * 60)
    print("ADIM 1: Manuel set toplanıyor...")
    manuel_set = set()
    for base in [TUNE_DIR, LOCK_DIR]:
        for sub in base.iterdir():
            if sub.is_dir():
                for f in sub.iterdir():
                    if f.suffix.lower() in IMAGE_EXTS:
                        manuel_set.add(f.name)
    print(f"  Manuel set: {len(manuel_set)} dosya")

    # ── ADIM 2: Positive → classify → raw ───────────────────────────────────
    print("\nADIM 2: Positive siniflandiriliyor...")
    pos_files = sorted([f for f in POSITIVE_DIR.iterdir()
                        if f.suffix.lower() in IMAGE_EXTS and f.name not in manuel_set])
    print(f"  Islenecek: {len(pos_files)} resim (manuel_set haric)")

    counts    = defaultdict(int)
    uncertain = 0
    total_pos = len(pos_files)

    for i, f in enumerate(pos_files, 1):
        label = classify_v1(f)
        if label == "uncertain":
            uncertain += 1
            continue
        shutil.copy2(f, RAW_DIRS[label] / f.name)
        counts[label] += 1
        if i % 2000 == 0 or i == total_pos:
            print(f"  [{i:>6}/{total_pos}]  "
                  f"long:{counts['longitudinal']:>5}  "
                  f"trans:{counts['transverse']:>5}  "
                  f"cross:{counts['cross']:>5}  "
                  f"uncertain:{uncertain:>4}")

    # ── ADIM 3: Negative → no_crack ─────────────────────────────────────────
    print("\nADIM 3: Negative kopyalaniyor...")
    neg_files = sorted([f for f in NEGATIVE_DIR.iterdir()
                        if f.suffix.lower() in IMAGE_EXTS])
    for i, f in enumerate(neg_files, 1):
        shutil.copy2(f, RAW_DIRS["no_crack"] / f.name)
        if i % 5000 == 0 or i == len(neg_files):
            print(f"  [{i:>6}/{len(neg_files)}]")
    counts["no_crack"] = len(neg_files)

    # ── ADIM 4: Raw sayıları ─────────────────────────────────────────────────
    print("\nADIM 4: RAW SINIF SAYILARI:")
    print(f"  {'SINIF':<20} {'SAYI':>7}")
    print("  " + "-" * 29)
    for c in CLASSES:
        print(f"  {c:<20} {counts[c]:>7}")
    print(f"  {'uncertain (atlandi)':<20} {uncertain:>7}")
    print(f"  {'TOPLAM':<20} {sum(counts.values()):>7}")

    # ── ADIM 5: 85/15 train/val split ───────────────────────────────────────
    print("\nADIM 5: Train/val split (85/15, seed=42)...")
    rng = random.Random(SEED)
    split_counts = {}

    for c in CLASSES:
        files = sorted([f for f in RAW_DIRS[c].iterdir()
                        if f.suffix.lower() in IMAGE_EXTS])
        rng.shuffle(files)
        n_val   = max(1, round(len(files) * 0.15))
        val_f   = files[:n_val]
        train_f = files[n_val:]
        for f in train_f:
            shutil.copy2(f, TRAIN_DIRS[c] / f.name)
        for f in val_f:
            shutil.copy2(f, VAL_DIRS[c] / f.name)
        split_counts[c] = (len(train_f), len(val_f))
        print(f"  {c:<20}: train={len(train_f):>5}, val={len(val_f):>5}")

    # ── ADIM 6: Augmentation → train/longitudinal ────────────────────────────
    print(f"\nADIM 6: Augmentation (train/longitudinal -> ~{AUG_TARGET})...")
    long_train = TRAIN_DIRS["longitudinal"]
    orig_files = sorted([f for f in long_train.iterdir()
                         if f.suffix.lower() in IMAGE_EXTS])
    n_orig   = len(orig_files)
    n_needed = max(0, AUG_TARGET - n_orig)
    print(f"  Mevcut: {n_orig}, hedef: {AUG_TARGET}, uretilecek: {n_needed}")

    aug_rng   = random.Random(SEED)
    aug_count = 0

    if n_needed > 0:
        aug_per_img = math.ceil(n_needed / n_orig)
        for f in orig_files:
            if aug_count >= n_needed:
                break
            buf = np.fromfile(str(f), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            for j in range(aug_per_img):
                if aug_count >= n_needed:
                    break
                aug_img  = augment_img(img, aug_rng)
                out_name = f"aug{j}_{f.stem}.jpg"
                cv2.imwrite(str(long_train / out_name), aug_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                aug_count += 1
        print(f"  Uretilen: {aug_count} augmented resim")
    else:
        print("  Augmentation gerekmiyor, hedef zaten karsilandi.")

    # ── ADIM 7: Final sayılar ────────────────────────────────────────────────
    print("\nADIM 7: FINAL SAYILAR:")
    print(f"  {'SINIF':<20} {'TRAIN':>8} {'VAL':>8} {'TOPLAM':>8}")
    print("  " + "-" * 48)
    total_train = total_val = 0
    for c in CLASSES:
        n_tr = len([f for f in TRAIN_DIRS[c].iterdir() if f.suffix.lower() in IMAGE_EXTS])
        n_vl = len([f for f in VAL_DIRS[c].iterdir()   if f.suffix.lower() in IMAGE_EXTS])
        total_train += n_tr
        total_val   += n_vl
        print(f"  {c:<20} {n_tr:>8} {n_vl:>8} {n_tr+n_vl:>8}")
    print("  " + "-" * 48)
    print(f"  {'TOPLAM':<20} {total_train:>8} {total_val:>8} {total_train+total_val:>8}")
    print("\nPipeline tamamlandi.")


if __name__ == "__main__":
    main()
