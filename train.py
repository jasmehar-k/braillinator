import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from unet_model import CharTokenizer, ConditionalUNet

CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
IMAGE_SIZE = 256


class OCRDataset(Dataset):
    def __init__(self, pair_dirs: list, tokenizer: CharTokenizer):
        self.pair_dirs = pair_dirs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pair_dirs)

    def __getitem__(self, idx: int):
        d = self.pair_dirs[idx]

        degraded = np.array(Image.open(os.path.join(d, "degraded.png")).convert("L"))
        clean = np.array(Image.open(os.path.join(d, "clean.png")).convert("L"))

        # Resize to 256x256
        if degraded.shape != (IMAGE_SIZE, IMAGE_SIZE):
            degraded = np.array(
                Image.fromarray(degraded).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            )
        if clean.shape != (IMAGE_SIZE, IMAGE_SIZE):
            clean = np.array(
                Image.fromarray(clean).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            )

        ocr_text = ""
        ocr_path = os.path.join(d, "ocr.txt")
        if os.path.exists(ocr_path):
            with open(ocr_path, encoding="utf-8") as f:
                ocr_text = f.read()

        degraded_t = torch.from_numpy(degraded.astype(np.float32) / 255.0).unsqueeze(0)
        clean_t = torch.from_numpy(clean.astype(np.float32) / 255.0).unsqueeze(0)

        tokens = self.tokenizer.encode(ocr_text)
        return degraded_t, tokens, clean_t


class MultiScaleL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = [1.0, 0.5, 0.25]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(pred, target) * self.weights[0]
        p, t = pred, target
        for w in self.weights[1:]:
            p = F.interpolate(p, scale_factor=0.5, mode="bilinear", align_corners=False)
            t = F.interpolate(t, scale_factor=0.5, mode="bilinear", align_corners=False)
            loss = loss + F.l1_loss(p, t) * w
        return loss


class CombinedLoss(nn.Module):
    """
    Multi-scale L1 + gradient loss for raw grayscale image restoration.

    The model uses residual learning so the training target is the clean raw
    grayscale image (not a binary image), making BCE inappropriate.
    Multi-scale L1 captures coarse structure at multiple resolutions;
    gradient loss penalises blurry edges in text strokes.
    """
    def __init__(self):
        super().__init__()
        self.ms_l1 = MultiScaleL1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1   = self.ms_l1(pred, target)
        grad = gradient_loss(pred, target)
        return l1 + 0.2 * grad


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch: int, val_loss: float, tokenizer: CharTokenizer, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "tokenizer_vocab": tokenizer.char_to_idx,
        },
        path,
    )


def load_checkpoint(model, optimizer, path: str):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["val_loss"]


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for degraded, tokens, clean in loader:
            degraded = degraded.to(device)
            tokens = tokens.to(device)
            clean = clean.to(device)

            pred = model(degraded, tokens)
            loss = loss_fn(pred, clean)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item()
    return total / max(len(loader), 1)


def train(dataset_dir: str = "dataset", num_epochs: int = NUM_EPOCHS) -> None:
    pairs_root = os.path.join(dataset_dir, "pairs")
    if not os.path.isdir(pairs_root):
        raise FileNotFoundError(f"Dataset not found at {pairs_root}. Run generate_training_data.py first.")

    all_dirs = sorted(
        os.path.join(pairs_root, d)
        for d in os.listdir(pairs_root)
        if os.path.isdir(os.path.join(pairs_root, d))
    )
    if not all_dirs:
        raise ValueError("No pairs found in dataset.")

    tokenizer = CharTokenizer()
    dataset = OCRDataset(all_dirs, tokenizer)

    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = get_device()
    print(f"Device: {device}  |  Train: {n_train}  |  Val: {n_val}")

    model = ConditionalUNet().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CombinedLoss().to(device)

    best_val = float("inf")
    start_epoch = 0

    if os.path.exists(BEST_CHECKPOINT):
        start_epoch, best_val = load_checkpoint(model, optimizer, BEST_CHECKPOINT)
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val:.4f}")

    for epoch in range(start_epoch, num_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        print(f"Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch + 1, best_val, tokenizer, BEST_CHECKPOINT)
            print(f"  Saved best checkpoint (val={best_val:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the OCR-conditioned U-Net.")
    parser.add_argument("--dataset", default="dataset", help="Dataset folder (default: dataset)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Training epochs (default: {NUM_EPOCHS})")
    args = parser.parse_args()
    train(args.dataset, args.epochs)
