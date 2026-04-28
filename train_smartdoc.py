"""
Train the ConditionalUNet on SmartDoc real phone-document photos using TrOCR
as a differentiable OCR loss.

No synthetic degradation — real phone photos are the input. The model is trained
so that its enhanced output is more readable by TrOCR, using Tesseract's reading
of the original patch as the pseudo ground-truth text target.

Usage:
    .venv/bin/python train_smartdoc.py \
        --images smartdoc_download/extracted/sampleDataset/input_sample \
        --gt     smartdoc_download/extracted/sampleDataset/input_sample_groundtruth \
        --epochs 60
"""

import argparse
import os
import random
import sys
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytesseract
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from unet_model import CharTokenizer, ConditionalUNet

warnings.filterwarnings("ignore")

CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
MIN_WORDS_PER_PATCH = 3    # skip patches where OCR finds fewer words than this
TROCR_INPUT_SIZE = 384
IDENTITY_WEIGHT = 0.3      # weight on L1(enhanced, input) * conf_map term
TROCR_WEIGHT = 1.0


def _conf_map(gray: np.ndarray) -> np.ndarray:
    """Return float32 OCR confidence heatmap [0, 1] for a grayscale patch."""
    conf = np.zeros(gray.shape, dtype=np.float32)
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        for i, c in enumerate(data["conf"]):
            if c == -1:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if w > 0 and h > 0:
                x2 = min(x + w, gray.shape[1])
                y2 = min(y + h, gray.shape[0])
                conf[max(0, y):y2, max(0, x):x2] = float(c) / 100.0
    except Exception:
        pass
    return conf


class SmartDocDataset(Dataset):
    """Loads pre-processed patches from preprocess_smartdoc.py output."""

    def __init__(self, dataset_dir: str, tokenizer: CharTokenizer):
        self.tokenizer = tokenizer
        pairs_root = os.path.join(dataset_dir, "pairs")
        self.pair_dirs = sorted(
            os.path.join(pairs_root, d)
            for d in os.listdir(pairs_root)
            if os.path.isdir(os.path.join(pairs_root, d))
        )
        if not self.pair_dirs:
            raise ValueError(f"No pairs found in {pairs_root}. Run preprocess_smartdoc.py first.")

    def __len__(self) -> int:
        return len(self.pair_dirs)

    def __getitem__(self, idx: int):
        d = self.pair_dirs[idx]

        patch = np.array(Image.open(os.path.join(d, "patch.png")).convert("L"))

        # Conditioning signal: Tesseract's reading (guides the FiLM layers)
        ocr_text = ""
        ocr_path = os.path.join(d, "ocr.txt")
        if os.path.exists(ocr_path):
            with open(ocr_path, encoding="utf-8", errors="ignore") as f:
                ocr_text = f.read().strip()

        # TrOCR supervision target: aligned document ground truth for this patch
        aligned_gt = ""
        ag_path = os.path.join(d, "aligned_gt.txt")
        if os.path.exists(ag_path):
            with open(ag_path, encoding="utf-8", errors="ignore") as f:
                aligned_gt = f.read().strip()
        if not aligned_gt:
            aligned_gt = ocr_text  # fallback if alignment failed

        conf_path = os.path.join(d, "conf.npy")
        conf = np.load(conf_path).astype(np.float32) if os.path.exists(conf_path) else \
               np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        img_t = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        conf_t = torch.from_numpy(conf).unsqueeze(0)
        input_t = torch.cat([img_t, conf_t], dim=0)   # (2, 256, 256)

        tokens = self.tokenizer.encode(ocr_text)  # conditioning

        return input_t, tokens, aligned_gt   # aligned_gt = TrOCR supervision target


def collate_fn(batch):
    inputs, tokens_list, ocr_texts = zip(*batch)
    inputs = torch.stack(inputs)
    tokens = torch.stack(tokens_list)
    return inputs, tokens, list(ocr_texts)


class TrOCRLoss(nn.Module):
    """Frozen TrOCR used as a differentiable OCR quality signal."""

    def __init__(self, device: torch.device):
        super().__init__()
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        print("[TrOCR] loading microsoft/trocr-base-printed...")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # Always keep TrOCR on CPU — MPS backward through ViT attention is very slow.
        # Gradients flow back to MPS automatically through the device-transfer op.
        self.model = self.model.to("cpu")
        self._unet_device = device
        print("[TrOCR] loaded (on cpu).")

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 256, 256) grayscale in [0, 1], any device
        # Move to CPU first (TrOCR lives on CPU); .cpu() is differentiable —
        # gradients flow back to the original device automatically.
        x = x.cpu()
        x = F.interpolate(x, size=(TROCR_INPUT_SIZE, TROCR_INPUT_SIZE),
                          mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        x = (x - 0.5) / 0.5
        return x

    def forward(self, enhanced: torch.Tensor, target_texts: list) -> torch.Tensor:
        """
        Compute cross-entropy OCR loss on CPU.
        Gradients flow back through .cpu() to the U-Net on MPS.
        """
        valid = [(i, t) for i, t in enumerate(target_texts) if t.strip()]
        if not valid:
            return torch.tensor(0.0, requires_grad=True)

        indices = [i for i, _ in valid]
        texts = [t for _, t in valid]

        enc_in = self.processor.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=64
        ).input_ids  # on CPU
        enc_in[enc_in == self.processor.tokenizer.pad_token_id] = -100

        pixel_values = self._preprocess(enhanced[indices])  # differentiable, on CPU

        out = self.model(pixel_values=pixel_values, labels=enc_in)
        return out.loss  # scalar on CPU


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, val_loss, tokenizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "tokenizer_vocab": tokenizer.char_to_idx,
    }, path)


def run_epoch(model, loader, optimizer, trocr_loss_fn, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for input_t, tokens, ocr_texts in loader:
            input_t = input_t.to(device)
            tokens = tokens.to(device)

            enhanced = model(input_t, tokens)   # (B, 1, 256, 256)

            # OCR quality loss: TrOCR should read the same text from enhanced as
            # Tesseract read from the original (pseudo ground-truth).
            trocr_loss = trocr_loss_fn(enhanced, ocr_texts)  # scalar on CPU

            # Identity regularization: don't distort regions where OCR was confident.
            conf_map = input_t[:, 1:2]
            orig_img = input_t[:, :1]
            identity_loss = F.l1_loss(enhanced * conf_map, orig_img * conf_map)  # on device

            # Combine: move trocr_loss to U-Net device; .to() is differentiable so
            # gradients still flow back through CPU TrOCR on backward.
            loss = TROCR_WEIGHT * trocr_loss.to(device) + IDENTITY_WEIGHT * identity_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients — TrOCR signal can produce large grads
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total += loss.item()
            n += 1

    return total / max(n, 1)


def train(dataset_dir: str, num_epochs: int) -> None:
    device = get_device()
    print(f"Device: {device}", flush=True)

    tokenizer = CharTokenizer()

    dataset = SmartDocDataset(dataset_dir, tokenizer)
    print(f"Dataset: {len(dataset)} patches from {dataset_dir}", flush=True)

    n_train = int(len(dataset) * 0.85)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    model = ConditionalUNet().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    trocr_loss_fn = TrOCRLoss(device)

    best_val = float("inf")
    start_epoch = 0

    if os.path.exists(BEST_CHECKPOINT):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val = ckpt["val_loss"]
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}, best val {best_val:.4f}")

    for epoch in range(start_epoch, num_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, trocr_loss_fn, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, trocr_loss_fn, device, train=False)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}",
              flush=True)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch + 1, best_val, tokenizer, BEST_CHECKPOINT)
            print(f"  Saved best checkpoint (val={best_val:.4f})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_smartdoc",
                        help="Pre-processed dataset dir from preprocess_smartdoc.py")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    train(args.dataset, args.epochs)
