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
