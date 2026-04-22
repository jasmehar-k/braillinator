# Braillinator
(v1 cloned from GitLab)

## Description
The Braillinator is a text from image to Braille converter that scans text using a phone camera and translates to Braille. The translation is done by pistons that automatically push out of a corresponding physical device, where the user can place their fingers on. These pistons push in and out, spelling out the text in Braille, letter by letter, giving the same tactile experience as Braille. This physical device also has play/pause, speed adjustments, and other controls.

## Team Members
Iya Sandhu - isandhu<br />
Hannah Wiens - h2wiens<br />
Jasmehar Kaur - j256kaur<br />
Kaniesa Deswal - k2deswal

## Course
SE101 Fall 2024

---

## v2: OCR Enhancement (In Progress)

### The Problem
The original pipeline relies on Tesseract OCR, which performs poorly on blurry or low-quality phone camera images. Since Braillinator is designed for visually impaired users, OCR errors directly result in incorrect Braille output — making reliability critical.

### What We Explored: Diffusion-Based Text Reconstruction
We investigated using a diffusion model conditioned on noisy OCR output to reconstruct sharp, legible text images before running final OCR. The approach: run a first-pass OCR to get a rough semantic signal, then feed that signal as conditioning into a diffusion model during denoising, so the reconstructed image is guided by what characters are likely present.

### Why Full Diffusion Doesn't Work Here
Stable Diffusion and similar full diffusion models are not viable for this use case for two reasons. First, inference on CPU (with no GPU) takes 30–90 seconds per image — too slow for a device where a user is standing and waiting for output. Second, these models are designed for text-to-image generation, not targeted image restoration, making them a poor architectural fit. Running any of this on a Raspberry Pi would make it worse by orders of magnitude.

### What We Built Instead
We're implementing a small conditional U-Net trained from scratch — architecturally capturing the core insight (OCR output conditions the image reconstruction) without the cost of iterative diffusion. The model takes two inputs: the blurry image and the Tesseract character predictions as a conditioning signal via FiLM (feature-wise linear modulation). One forward pass produces a sharpened image, which is then passed to Tesseract for a final, more accurate read. The model is designed to be under 50MB and fast enough for inference on a Raspberry Pi 4.

Training data is generated synthetically: clean printed text images are degraded with realistic blur, noise, and compression artifacts, and the model is trained to restore them using the paired OCR predictions as conditioning.

---

## v2 Progress Log

### Round 1 — baseline U-Net (overfitting on 16 images)
First version trained on 16 real photos from the project. Val loss plateaued at ~0.12 and results on new images were terrible — the model had basically memorized the training set instead of learning anything general. Not surprising in hindsight, 16 images is nowhere near enough.

### Round 2 — synthetic training data + training fixes
Generated 200 synthetic printed-text pages using system fonts (Times, Helvetica, Courier, etc.) and degraded them to create 5000 training pairs. Also fixed two bugs found during this pass:
- **double gradient loss**: the loss function was applying gradient penalty twice, once inside `CombinedLoss` and again in `run_epoch`
- **no LR scheduler**: training used a flat learning rate the whole time, added CosineAnnealingLR

Also bumped the residual scale from 0.3 → 0.5 so the model could make stronger corrections on heavily degraded images.

Val loss improved from ~0.12 to ~0.02. Looked promising on paper.

### Why Round 2 still didn't work on real images
Evaluation against the actual test images showed the enhancement was making things *worse* — the enhanced output was more garbled than the baseline OCR. The issue was domain gap: the model trained entirely on clean font renders degraded synthetically, and real phone camera photos look completely different. The degradation distribution didn't match.

### Round 3 — adding real images (TextOCR dataset)
Downloaded the TextOCR dataset (~7GB, 25k real-world photos of text). Sampled 800 of those and mixed them with the 200 synthetic images, regenerated 5000 training pairs, retrained from scratch.

Average misspelling rate dropped from 12.8% → 9.2% on test images. Two images saw real improvement:
- TestImage_1 (heavily blurred): 85.7% → 33.6% miss rate
- TestImage_4: 30.6% → 25.3% miss rate

Everything else correctly skipped by the quality gate (only enhances images where ≥10 words are detected AND miss rate >25%).

But visually the enhanced output on the problem images was still generating noise/dashes rather than readable text. The metric improved because dashes aren't counted as misspelled words — so the numbers looked better than they actually were.

### What we looked at next — better datasets
Looked into datasets with real paired (degraded, clean) document images so we wouldn't need synthetic degradation at all. Turns out almost nothing exists at scale. DocUNet has 130 pairs, NoisyOffice has 72. That's not enough to train a U-Net properly. Dead end.

### Round 4 (current) — fixing the architecture itself
The real problem wasn't just the data — it was that the architecture was using the OCR signal badly:

1. **FiLM only at the bottleneck**: text conditioning only touched the deepest 16×16 features. By the time the decoder reached full resolution, the signal had been diluted through skip connections that were never conditioned.

2. **Mean pooling over character embeddings**: "hello world" and "world hello" produced identical conditioning vectors. Word order and structure were completely lost.

3. **No spatial grounding**: the model knew "this image contains some text" but had no idea *where* in the image the OCR was struggling.

Current fix being trained now:
- **Spatial confidence map as a second input channel**: Tesseract's `image_to_data()` returns per-word confidence scores. We render these as a heatmap and feed it alongside the image — the model now sees exactly which regions OCR is uncertain about.
- **Multi-level FiLM**: text conditioning applied at every decoder stage (32×32, 64×64, 128×128) not just the bottleneck, so the semantic signal stays present all the way to full resolution.

---

## Acknowledgements

This project uses the following open-source libraries:

* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (Apache License 2.0)
* [NumPy](https://numpy.org/) (BSD License)
* [OpenCV](https://opencv.org/) (Apache License 2.0)
* [Pillow](https://pillow.readthedocs.io/en/stable/) (PIL Software License)
* [pyspellchecker](https://github.com/bagder/pyspellchecker) (MIT License)
* [Levenshtein](https://github.com/mikekap/levenshtein) (MIT License)
* [RPi.GPIO](https://pypi.org/project/RPi.GPIO/) (MIT License)
* [PyTorch](https://pytorch.org/) (BSD License)
