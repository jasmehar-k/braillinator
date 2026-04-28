"""
Two evaluation modes:

1. Default (UsedImages sanity check):
   .venv/bin/python evaluate.py
   Miss rate + word-count-aware effective delta on the 16 test images.

2. SmartDoc held-out WER (primary metric):
   .venv/bin/python evaluate.py --smartdoc \
     --images smartdoc_download/extracted/sampleDataset/input_sample \
     --gt     smartdoc_download/extracted/sampleDataset/input_sample_groundtruth \
     --split-start 2905
   WER against real ground-truth on held-out images (never seen during training).
"""
import argparse
import os

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from spellchecker import SpellChecker

import enhance
import sharpenImage

spell = SpellChecker()

WORD_COUNT_FLOOR = 0.5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def ocr(img: np.ndarray) -> str:
    return pytesseract.image_to_string(img, lang="eng") or ""


def word_count(text: str) -> int:
    return len([w for w in text.split() if w.isalpha()])


def misspell_rate(text: str) -> float:
    words = [w.strip(".,;:!?\"'()[]") for w in text.split()]
    words = [w for w in words if w.isalpha() and len(w) > 1]
    if not words:
        return 0.0
    return len(spell.unknown(words)) / len(words)


def wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate: edit distance at word level / reference word count."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0
    # Dynamic programming edit distance
    d = list(range(len(hyp) + 1))
    for r in ref:
        prev = d[0]
        d[0] += 1
        for j, h in enumerate(hyp):
            temp = d[j + 1]
            d[j + 1] = prev if r == h else 1 + min(prev, d[j], d[j + 1])
            prev = temp
    return d[len(hyp)] / len(ref)


# ---------------------------------------------------------------------------
# Mode 1: UsedImages miss-rate evaluation
# ---------------------------------------------------------------------------

def evaluate_used_images(images_dir: str) -> None:
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print("No images found.")
        return

    print(f"{'Image':<18} {'Base':>5} {'Miss%':>6}  {'Enh':>5} {'Miss%':>6}  {'ΔMiss':>7}  {'ΔWords':>7}  {'Result'}")
    print("-" * 80)

    total_base, total_eff, n = 0.0, 0.0, 0

    for fname in files:
        path = os.path.join(images_dir, fname)
        preprocessed = sharpenImage.preprocessing(path)
        base_text = ocr(preprocessed)
        base_miss = misspell_rate(base_text)
        base_wc = word_count(base_text)

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
            enhanced_pre = sharpenImage.preprocess_array(enhanced_raw)
            enh_text = ocr(enhanced_pre)
        else:
            enh_text = base_text

        enh_miss = misspell_rate(enh_text)
        enh_wc = word_count(enh_text)
        delta_miss = enh_miss - base_miss
        delta_words = enh_wc - base_wc

        word_count_ok = (base_wc == 0) or (enh_wc >= base_wc * WORD_COUNT_FLOOR)
        if gate_passed and not word_count_ok:
            effective_miss = base_miss
            result = "NOISE (words collapsed)"
        elif not gate_passed:
            effective_miss = base_miss
            result = "skipped (gate)"
        elif delta_miss < -0.05:
            effective_miss = enh_miss
            result = "better"
        elif delta_miss > 0.05:
            effective_miss = enh_miss
            result = "worse"
        else:
            effective_miss = enh_miss
            result = "no change"

        print(f"{fname:<18} {base_wc:>5} {base_miss*100:>5.1f}%  {enh_wc:>5} {enh_miss*100:>5.1f}%  {delta_miss*100:>+6.1f}%  {delta_words:>+6}  {result}")
        total_base += base_miss
        total_eff += effective_miss
        n += 1

    if n:
        print("-" * 80)
        avg_base = total_base / n * 100
        avg_eff = total_eff / n * 100
        print(f"{'AVERAGE':<18} {'':>5} {avg_base:>5.1f}%  {'':>5} {'':>5}   {avg_eff - avg_base:>+6.1f}%  {'':>6}  (effective delta)")
        print("Note: effective delta only counts improvements where word count didn't collapse.")


# ---------------------------------------------------------------------------
# Mode 2: SmartDoc held-out WER evaluation
# ---------------------------------------------------------------------------

def evaluate_smartdoc(images_dir: str, gt_dir: str, split_start: int, max_images: int) -> None:
    fnames = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".jpg"))
    # held-out = images from split_start onward (1-indexed filenames)
    test_fnames = [f for f in fnames if int(os.path.splitext(f)[0]) >= split_start]
    if max_images:
        test_fnames = test_fnames[:max_images]

    if not test_fnames:
        print(f"No held-out images found with index >= {split_start}")
        return

    print(f"SmartDoc held-out evaluation: {len(test_fnames)} images (index >= {split_start})")
    print(f"{'Image':<14} {'Base WER':>9}  {'Enh WER':>8}  {'ΔWER':>7}  {'Result'}")
    print("-" * 60)

    total_base_wer, total_enh_wer, n = 0.0, 0.0, 0

    for fname in test_fnames:
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(images_dir, fname)
        gt_path = os.path.join(gt_dir, stem + ".txt")
        if not os.path.exists(gt_path):
            continue

        with open(gt_path, encoding="utf-8", errors="ignore") as f:
            gt_text = f.read().strip()

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        base_text = ocr(img)
        base_wer = wer(base_text, gt_text)

        # Quality gate
        base_miss = misspell_rate(base_text)
        gate_passed = len(base_text.split()) >= 10 and base_miss > 0.25

        if gate_passed:
            enhanced = enhance.enhance_image(img, base_text)
            enh_text = ocr(enhanced)
        else:
            enh_text = base_text

        enh_wer_val = wer(enh_text, gt_text)
        delta = enh_wer_val - base_wer

        if not gate_passed:
            result = "skipped (gate)"
        elif delta < -0.02:
            result = "better"
        elif delta > 0.02:
            result = "worse"
        else:
            result = "no change"

        print(f"{fname:<14} {base_wer*100:>8.1f}%  {enh_wer_val*100:>7.1f}%  {delta*100:>+6.1f}%  {result}")
        total_base_wer += base_wer
        total_enh_wer += enh_wer_val
        n += 1

    if n:
        print("-" * 60)
        print(f"{'AVERAGE':<14} {total_base_wer/n*100:>8.1f}%  {total_enh_wer/n*100:>7.1f}%  {(total_enh_wer-total_base_wer)/n*100:>+6.1f}%")
        print(f"\nWER = word edit distance / ground truth word count (lower is better)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smartdoc", action="store_true", help="Run SmartDoc held-out WER evaluation")
    parser.add_argument("--images", default="smartdoc_download/extracted/sampleDataset/input_sample")
    parser.add_argument("--gt", default="smartdoc_download/extracted/sampleDataset/input_sample_groundtruth")
    parser.add_argument("--split-start", type=int, default=2905,
                        help="First image index (filename) to include in held-out test set")
    parser.add_argument("--max-images", type=int, default=50,
                        help="Max held-out images to evaluate (default 50 for speed)")
    parser.add_argument("--used-images", default="UsedImages")
    args = parser.parse_args()

    if args.smartdoc:
        evaluate_smartdoc(args.images, args.gt, args.split_start, args.max_images)
    else:
        evaluate_used_images(args.used_images)
