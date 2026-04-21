"""
Compare baseline OCR vs enhancement-then-OCR on all test images.
Prints per-image word counts and a misspelling rate summary.
"""
import os
import sys

import cv2
import numpy as np
import pytesseract
from spellchecker import SpellChecker

import enhance
import sharpenImage

spell = SpellChecker()


def ocr(img: np.ndarray) -> str:
    return pytesseract.image_to_string(img, lang="eng") or ""


def misspell_rate(text: str) -> float:
    words = [w.strip(".,;:!?\"'()[]") for w in text.split()]
    words = [w for w in words if w.isalpha() and len(w) > 1]
    if not words:
        return 0.0
    bad = spell.unknown(words)
    return len(bad) / len(words)


def word_count(text: str) -> int:
    return len([w for w in text.split() if w.isalpha()])


def evaluate(images_dir: str = "UsedImages") -> None:
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print("No images found.")
        return

    print(f"{'Image':<18} {'Base words':>10} {'Base miss%':>10} {'Enh words':>10} {'Enh miss%':>10} {'Delta miss%':>11}")
    print("-" * 75)

    total_base_miss, total_enh_miss, n = 0.0, 0.0, 0

    for fname in files:
        path = os.path.join(images_dir, fname)

        # Baseline: preprocess → OCR
        preprocessed = sharpenImage.preprocessing(path)
        base_text = ocr(preprocessed)
        base_miss = misspell_rate(base_text)
        base_words = word_count(base_text)

        # Enhanced: respect the same quality gate as imageToText.py
        # Only run U-Net when initial OCR has ≥10 words and fails quality check
        from PIL import Image, ImageOps
        enough_words = len(base_text.split()) >= 10
        gate_passed = enough_words and (base_miss > 0.25)
        if gate_passed:
            pil = Image.open(path)
            try:
                pil = ImageOps.exif_transpose(pil)
            except Exception:
                pass
            raw = np.array(pil.convert("L"))
            enhanced_raw = enhance.enhance_image(raw, base_text)
            enhanced_preprocessed = sharpenImage.preprocess_array(enhanced_raw)
            enh_text = ocr(enhanced_preprocessed)
        else:
            enh_text = base_text  # quality gate skipped enhancement
        enh_miss = misspell_rate(enh_text)
        enh_words = word_count(enh_text)

        delta = enh_miss - base_miss
        marker = " <-- worse" if delta > 0.05 else (" <-- better" if delta < -0.05 else "")
        print(f"{fname:<18} {base_words:>10} {base_miss*100:>9.1f}% {enh_words:>10} {enh_miss*100:>9.1f}% {delta*100:>+10.1f}%{marker}")

        total_base_miss += base_miss
        total_enh_miss += enh_miss
        n += 1

    if n:
        avg_base = total_base_miss / n * 100
        avg_enh = total_enh_miss / n * 100
        print("-" * 75)
        print(f"{'AVERAGE':<18} {'':>10} {avg_base:>9.1f}% {'':>10} {avg_enh:>9.1f}% {avg_enh-avg_base:>+10.1f}%")


if __name__ == "__main__":
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "UsedImages"
    evaluate(img_dir)
