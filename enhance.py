import os

import cv2
import numpy as np
import torch

from unet_model import CharTokenizer, ConditionalUNet

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt")
IMAGE_SIZE = 256
TILE_OVERLAP = 64   # pixels of overlap between adjacent tiles for smooth blending

# Module-level cache — model is loaded once per process
_model: ConditionalUNet = None
_tokenizer: CharTokenizer = None
_device: torch.device = None


def _load_model():
    global _model, _tokenizer, _device

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    tokenizer = CharTokenizer(char_to_idx=ckpt.get("tokenizer_vocab"))
    model = ConditionalUNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        model = model.to(device)
        dummy_x = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        dummy_tok = torch.zeros(1, 64, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(dummy_x, dummy_tok)
        if out.isnan().any():
            raise RuntimeError("NaN in warm-up output")
    except RuntimeError:
        device = torch.device("cpu")
        model = model.to(device)

    _model = model
    _tokenizer = tokenizer
    _device = device


def _run_patch(patch: np.ndarray, tokens: torch.Tensor) -> np.ndarray:
    """Run a single 256×256 uint8 patch through the model. Returns uint8 patch."""
    t = torch.from_numpy(patch.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0).to(_device)
    with torch.no_grad():
        try:
            out = _model(t, tokens)
            if out.isnan().any():
                raise RuntimeError("NaN")
        except RuntimeError:
            _model.to("cpu")
            t = t.to("cpu")
            tokens = tokens.to("cpu")
            out = _model(t, tokens)
    return (out.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


def _enhance_tiled(image: np.ndarray, tokens: torch.Tensor) -> np.ndarray:
    """
    Tile the image into overlapping 256×256 patches, enhance each one, and
    blend back using a linear weight ramp so seams are invisible.

    Training used 256×256 patches extracted from full-resolution images so
    text characters appear at their native size.  Tiled inference matches
    this: each tile covers a small region of the page rather than a severely
    downscaled whole-page view.
    """
    h, w = image.shape
    stride = IMAGE_SIZE - TILE_OVERLAP

    # Accumulator and weight map for blending
    acc = np.zeros((h, w), dtype=np.float64)
    wts = np.zeros((h, w), dtype=np.float64)

    # 1-D raised-cosine weight (smooth at tile boundaries)
    ramp = np.hanning(IMAGE_SIZE).astype(np.float64)
    tile_weight = np.outer(ramp, ramp)

    # Collect tile top-left corners; ensure last tile touches the edge
    def tile_starts(length):
        starts = list(range(0, max(length - IMAGE_SIZE, 0) + 1, stride))
        if not starts or starts[-1] + IMAGE_SIZE < length:
            starts.append(max(0, length - IMAGE_SIZE))
        return starts

    for y0 in tile_starts(h):
        for x0 in tile_starts(w):
            y1, x1 = y0 + IMAGE_SIZE, x0 + IMAGE_SIZE
            # Pad if image is smaller than tile (shouldn't happen often)
            patch = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
            ph = min(IMAGE_SIZE, h - y0)
            pw = min(IMAGE_SIZE, w - x0)
            patch[:ph, :pw] = image[y0 : y0 + ph, x0 : x0 + pw]

            enhanced = _run_patch(patch, tokens)

            acc[y0 : y0 + ph, x0 : x0 + pw] += (
                enhanced[:ph, :pw].astype(np.float64) * tile_weight[:ph, :pw]
            )
            wts[y0 : y0 + ph, x0 : x0 + pw] += tile_weight[:ph, :pw]

    wts = np.maximum(wts, 1e-8)
    return np.clip(acc / wts, 0, 255).astype(np.uint8)


def enhance_image(image: np.ndarray, ocr_text: str) -> np.ndarray:
    """
    Apply U-Net restoration conditioned on OCR text.

    Args:
        image:    uint8 numpy array, grayscale (H, W) or (H, W, 1)
        ocr_text: string from an initial Tesseract pass; may be empty

    Returns:
        uint8 numpy array, same spatial shape as input (H, W).
        Returns image unchanged if checkpoints/best.pt does not exist.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        print("[enhance] No checkpoint found — returning image unchanged.")
        return image

    if _model is None:
        _load_model()

    # Ensure 2-D
    if image.ndim == 3:
        image = image[:, :, 0]

    orig_h, orig_w = image.shape
    tokens = _tokenizer.encode(ocr_text).unsqueeze(0).to(_device)

    if orig_h >= IMAGE_SIZE and orig_w >= IMAGE_SIZE:
        # Use tiled inference to process at native resolution.
        return _enhance_tiled(image, tokens)
    else:
        # Small image: resize to 256×256, enhance, resize back
        resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        result = _run_patch(resized, tokens)
        if (orig_h, orig_w) != (IMAGE_SIZE, IMAGE_SIZE):
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return result
