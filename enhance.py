import os

import cv2
import numpy as np
import torch

from unet_model import CharTokenizer, ConditionalUNet

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt")
IMAGE_SIZE = 256

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
        # Warm-up probe to catch silent MPS failures early
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

    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    t = torch.from_numpy(resized.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0).to(_device)          # (1,1,256,256)

    tokens = _tokenizer.encode(ocr_text).unsqueeze(0).to(_device)  # (1,64)

    with torch.no_grad():
        try:
            out = _model(t, tokens)
            if out.isnan().any():
                raise RuntimeError("NaN in model output")
        except RuntimeError:
            # MPS fallback to CPU
            _model.to("cpu")
            t = t.to("cpu")
            tokens = tokens.to("cpu")
            out = _model(t, tokens)

    result = out.squeeze().cpu().numpy()                   # (256,256) float32 in [0,1]
    result = (result * 255.0).clip(0, 255).astype(np.uint8)

    if (orig_h, orig_w) != (IMAGE_SIZE, IMAGE_SIZE):
        result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return result
