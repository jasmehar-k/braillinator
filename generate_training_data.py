import argparse
import os
import platform
import random

import cv2
import numpy as np
import pytesseract
from PIL import Image

import sharpenImage

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --- Degradation functions ---

def apply_gaussian_blur(img: np.ndarray) -> np.ndarray:
    sigma = random.uniform(1.0, 4.0)
    ksize = int(sigma * 6) | 1  # must be odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_motion_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([7, 11, 15])
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def apply_jpeg_compression(img: np.ndarray) -> np.ndarray:
    quality = random.randint(20, 60)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def apply_brightness_contrast_jitter(img: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.6, 1.4)
    beta = random.uniform(-40, 40)
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray) -> np.ndarray:
    std = random.uniform(5, 30)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


_DEGRADATIONS = [
    apply_gaussian_blur,
    apply_motion_blur,
    apply_jpeg_compression,
    apply_brightness_contrast_jitter,
    apply_gaussian_noise,
]


def degrade_image(img: np.ndarray) -> np.ndarray:
    fns = random.sample(_DEGRADATIONS, k=random.randint(2, 4))
    for fn in fns:
        img = fn(img)
    return img


def get_ocr_text(img: np.ndarray) -> str:
    try:
        text = pytesseract.image_to_string(img, lang="eng")
        return text if text else ""
    except Exception:
        return ""


# --- Dataset generation ---

def generate_dataset(
    clean_images_dir: str,
    output_dir: str = "dataset",
    target_pairs: int = 5000,
) -> None:
    src_files = sorted(
        f for f in os.listdir(clean_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not src_files:
        raise ValueError(f"No images found in {clean_images_dir}")

    # Split source images 80/20 by identity to avoid correlated train/val sets
    random.shuffle(src_files)
    split = max(1, int(len(src_files) * 0.8))
    train_files = src_files[:split]
    val_files = src_files[split:] or src_files[:1]  # guarantee at least 1 val image

    pairs_per_split = {
        "train": int(target_pairs * 0.8),
        "val": target_pairs - int(target_pairs * 0.8),
    }
    split_files = {"train": train_files, "val": val_files}

    pair_idx = 0
    for split_name, files in split_files.items():
        n_pairs = pairs_per_split[split_name]
        per_image = max(1, n_pairs // len(files))
        generated = 0

        for fname in files:
            path = os.path.join(clean_images_dir, fname)
            try:
                clean = sharpenImage.preprocessing(path)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
                continue

            for variant in range(per_image):
                if generated >= n_pairs:
                    break

                random.seed(pair_idx)
                np.random.seed(pair_idx)

                degraded = degrade_image(clean.copy())
                ocr_text = get_ocr_text(degraded)

                pair_dir = os.path.join(output_dir, "pairs", f"{pair_idx:05d}")
                os.makedirs(pair_dir, exist_ok=True)

                cv2.imwrite(os.path.join(pair_dir, "degraded.png"), degraded)
                cv2.imwrite(os.path.join(pair_dir, "clean.png"), clean)
                with open(os.path.join(pair_dir, "ocr.txt"), "w", encoding="utf-8") as f:
                    f.write(ocr_text)

                pair_idx += 1
                generated += 1

                if pair_idx % 100 == 0:
                    print(f"  [{split_name}] {generated}/{n_pairs} pairs")

        print(f"[{split_name}] Done: {generated} pairs")

    print(f"\nDataset written to {output_dir}/pairs/ ({pair_idx} total pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate (degraded, clean, ocr) training triplets for the U-Net."
    )
    parser.add_argument("--input", required=True, help="Folder of clean printed-text images")
    parser.add_argument("--output", default="dataset", help="Output dataset folder (default: dataset)")
    parser.add_argument("--count", type=int, default=5000, help="Target number of pairs (default: 5000)")
    args = parser.parse_args()

    generate_dataset(args.input, args.output, args.count)
