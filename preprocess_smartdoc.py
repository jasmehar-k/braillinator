"""
Extract 256x256 patches from SmartDoc phone-document photos.
Runs Tesseract once per patch and saves: patch.png, ocr.txt, conf.npy.
Output matches the format expected by train_smartdoc.py.

Usage:
    .venv/bin/python preprocess_smartdoc.py \
        --images smartdoc_download/extracted/sampleDataset/input_sample \
        --gt     smartdoc_download/extracted/sampleDataset/input_sample_groundtruth \
        --output dataset_smartdoc \
        --patches-per-image 4
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import pytesseract
from PIL import Image

IMAGE_SIZE = 256
MIN_WORDS = 3


def conf_map(gray: np.ndarray) -> np.ndarray:
    result = np.zeros(gray.shape, dtype=np.float32)
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        for i, c in enumerate(data["conf"]):
            if c == -1:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if w > 0 and h > 0:
                result[max(0,y):min(y+h, gray.shape[0]),
                       max(0,x):min(x+w, gray.shape[1])] = float(c) / 100.0
    except Exception:
        pass
    return result


def process_image(img_path: str, gt_path: str, out_dir: str, n_patches: int, idx_start: int) -> int:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    h, w = img.shape
    if h < IMAGE_SIZE or w < IMAGE_SIZE:
        scale = max(IMAGE_SIZE / h, IMAGE_SIZE / w) + 0.01
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape

    with open(gt_path, encoding="utf-8", errors="ignore") as f:
        gt_text = f.read().strip()

    saved = 0
    attempts = 0
    while saved < n_patches and attempts < n_patches * 6:
        attempts += 1
        x0 = random.randint(0, w - IMAGE_SIZE)
        y0 = random.randint(0, h - IMAGE_SIZE)
        patch = img[y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE]

        try:
            ocr_text = pytesseract.image_to_string(patch, config="--psm 6").strip()
        except Exception:
            ocr_text = ""

        if len(ocr_text.split()) < MIN_WORDS:
            continue

        cmap = conf_map(patch)

        pair_dir = os.path.join(out_dir, "pairs", f"{idx_start + saved:06d}")
        os.makedirs(pair_dir, exist_ok=True)

        cv2.imwrite(os.path.join(pair_dir, "patch.png"), patch)
        with open(os.path.join(pair_dir, "ocr.txt"), "w", encoding="utf-8") as f:
            f.write(ocr_text)
        with open(os.path.join(pair_dir, "gt.txt"), "w", encoding="utf-8") as f:
            f.write(gt_text)
        np.save(os.path.join(pair_dir, "conf.npy"), cmap)

        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="smartdoc_download/extracted/sampleDataset/input_sample")
    parser.add_argument("--gt", default="smartdoc_download/extracted/sampleDataset/input_sample_groundtruth")
    parser.add_argument("--output", default="dataset_smartdoc")
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    fnames = sorted(f for f in os.listdir(args.images) if f.lower().endswith(".jpg"))
    if args.max_images:
        fnames = fnames[:args.max_images]

    total_saved = 0
    for i, fname in enumerate(fnames):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(args.images, fname)
        gt_path = os.path.join(args.gt, stem + ".txt")
        if not os.path.exists(gt_path):
            continue

        n = process_image(img_path, gt_path, args.output, args.patches_per_image, total_saved)
        total_saved += n

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(fnames)}] {total_saved} patches saved", flush=True)

    print(f"Done. {total_saved} patches in {args.output}/pairs/", flush=True)


if __name__ == "__main__":
    main()
