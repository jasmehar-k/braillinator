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


def _generate_conf_map(image: np.ndarray) -> np.ndarray:
    """Return float32 confidence heatmap (1=OCR confident, 0=uncertain/no word)."""
    import pytesseract
    conf_map = np.zeros(image.shape[:2], dtype=np.float32)
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        for i, conf in enumerate(data["conf"]):
            if conf == -1:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if w > 0 and h > 0:
                x2 = min(x + w, image.shape[1])
                y2 = min(y + h, image.shape[0])
                conf_map[max(0, y):y2, max(0, x):x2] = float(conf) / 100.0
    except Exception:
        pass
    return conf_map


def _run_patch(patch: np.ndarray, conf_patch: np.ndarray, tokens: torch.Tensor) -> np.ndarray:
    """Run a single 256×256 patch through the model. Returns uint8 patch."""
    img_t = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    conf_t = torch.from_numpy(conf_patch).unsqueeze(0).unsqueeze(0)
    t = torch.cat([img_t, conf_t], dim=1).to(_device)  # (1, 2, H, W)
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


def _enhance_tiled(image: np.ndarray, conf_map: np.ndarray, tokens: torch.Tensor) -> np.ndarray:
    """
    Tile the image into overlapping 256×256 patches, enhance each one, and
    blend back using a linear weight ramp so seams are invisible.
    """
    h, w = image.shape
    stride = IMAGE_SIZE - TILE_OVERLAP

    acc = np.zeros((h, w), dtype=np.float64)
    wts = np.zeros((h, w), dtype=np.float64)

    ramp = np.hanning(IMAGE_SIZE).astype(np.float64)
    tile_weight = np.outer(ramp, ramp)

    def tile_starts(length):
        starts = list(range(0, max(length - IMAGE_SIZE, 0) + 1, stride))
        if not starts or starts[-1] + IMAGE_SIZE < length:
            starts.append(max(0, length - IMAGE_SIZE))
        return starts

    for y0 in tile_starts(h):
        for x0 in tile_starts(w):
            ph = min(IMAGE_SIZE, h - y0)
            pw = min(IMAGE_SIZE, w - x0)

            patch = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
            patch[:ph, :pw] = image[y0 : y0 + ph, x0 : x0 + pw]

            conf_patch = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            conf_patch[:ph, :pw] = conf_map[y0 : y0 + ph, x0 : x0 + pw]

            enhanced = _run_patch(patch, conf_patch, tokens)

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

    conf_map = _generate_conf_map(image)

    if orig_h >= IMAGE_SIZE and orig_w >= IMAGE_SIZE:
        return _enhance_tiled(image, conf_map, tokens)
    else:
        resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        conf_resized = cv2.resize(conf_map, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        result = _run_patch(resized, conf_resized, tokens)
        if (orig_h, orig_w) != (IMAGE_SIZE, IMAGE_SIZE):
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return result
