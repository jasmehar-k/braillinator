# v2 Progress Log

## Round 1 — baseline U-Net (overfitting on 16 images)
First version trained on 16 real photos from the project. Val loss plateaued at ~0.12 and results on new images were terrible — the model had basically memorized the training set instead of learning anything general. Not surprising in hindsight, 16 images is nowhere near enough.

## Round 2 — synthetic training data + training fixes
Generated 200 synthetic printed-text pages using system fonts (Times, Helvetica, Courier, etc.) and degraded them to create 5000 training pairs. Also fixed two bugs found during this pass:
- **double gradient loss**: the loss function was applying gradient penalty twice, once inside `CombinedLoss` and again in `run_epoch`
- **no LR scheduler**: training used a flat learning rate the whole time, added CosineAnnealingLR

Also bumped the residual scale from 0.3 → 0.5 so the model could make stronger corrections on heavily degraded images.

Val loss improved from ~0.12 to ~0.02. Looked promising on paper.

### Why Round 2 still didn't work on real images
Evaluation against the actual test images showed the enhancement was making things *worse* — the enhanced output was more garbled than the baseline OCR. The issue was domain gap: the model trained entirely on clean font renders degraded synthetically, and real phone camera photos look completely different. The degradation distribution didn't match.

## Round 3 — adding real images (TextOCR dataset)
Downloaded the TextOCR dataset (~7GB, 25k real-world photos of text). Sampled 800 of those and mixed them with the 200 synthetic images, regenerated 5000 training pairs, retrained from scratch.

Average misspelling rate dropped from 12.8% → 9.2% on test images. Two images saw real improvement:
- TestImage_1 (heavily blurred): 85.7% → 33.6% miss rate
- TestImage_4: 30.6% → 25.3% miss rate

Everything else correctly skipped by the quality gate (only enhances images where ≥10 words are detected AND miss rate >25%).

But visually the enhanced output on the problem images was still generating noise/dashes rather than readable text. The metric improved because dashes aren't counted as misspelled words — so the numbers looked better than they actually were.

### What we looked at next — better datasets
Looked into datasets with real paired (degraded, clean) document images so we wouldn't need synthetic degradation at all. Turns out almost nothing exists at scale. DocUNet has 130 pairs, NoisyOffice has 72. That's not enough to train a U-Net properly. Dead end.

## Round 4 — fixing the architecture
The real problem wasn't just the data — it was that the architecture was using the OCR signal badly:

1. **FiLM only at the bottleneck**: text conditioning only touched the deepest 16×16 features. By the time the decoder reached full resolution, the signal had been diluted through skip connections that were never conditioned.
2. **Mean pooling over character embeddings**: "hello world" and "world hello" produced identical conditioning vectors. Word order and structure were completely lost.
3. **No spatial grounding**: the model knew "this image contains some text" but had no idea *where* in the image the OCR was struggling.

Two architecture fixes were made:
- **Spatial confidence map as a second input channel**: Tesseract's `image_to_data()` returns per-word confidence scores. We render these as a heatmap and feed it alongside the image — the model now sees exactly which regions OCR is uncertain about.
- **Multi-level FiLM**: text conditioning applied at every decoder stage (32×32, 64×64, 128×128) not just the bottleneck, so the semantic signal stays present all the way to full resolution.

### Round 4 results — mixed
Val loss actually went up slightly (0.1217 → 0.1326) and average improvement was about the same (-3.1% vs -3.6%). TestImage_1 got a bit better but TestImage_4 regressed. The architecture changes are the right direction but they're not making a real difference yet — the training data is still the bottleneck. The model can't learn to use the confidence map properly when it's trained on synthetic degradation that doesn't match real camera photos.

## Round 5 — real phone photos + TrOCR loss (Tesseract pseudo-GT)
Switched the entire training approach. Instead of paired (degraded, clean) images, we used the SmartDoc ICDAR 2015 dataset — 3,630 real smartphone-captured document photos with ground truth text. No synthetic degradation at all.

The training loss is now differentiable OCR quality: we pass the U-Net's enhanced output through a frozen TrOCR model (microsoft/trocr-base-printed) on CPU, and minimize the cross-entropy between what TrOCR reads from the enhanced image vs. what Tesseract reads from the original. Gradients flow back through the TrOCR encoder to the U-Net. An identity regularization term (L1 weighted by the OCR confidence map) prevents the model from distorting regions where text was already readable.

Preprocessing extracted 14,520 patches from the 3,630 photos, ran Tesseract once per patch to cache OCR text and confidence maps, then trained for 10 epochs.

Val loss (TrOCR cross-entropy) dropped from 11.24 → 6.24 over 10 epochs, with the sharpest drop in epochs 3-4.

### Round 5 results
Average miss rate: 12.8% → 8.2% (-4.6%). The two problem images both improved:
- TestImage_1 (heavily blurred): 85.7% → 42.1% miss rate (-43.6%)
- TestImage_4: 30.6% → 0.0% miss rate (-30.6%)

Best result so far on TestImage_1. But the word count drop on TestImage_4 (30 words → 0 words) revealed a new problem: the model was generating patterns that drove miss rate to 0% by destroying all readable text. The metric looked good; the output was garbage. A word-count-aware NOISE flag was added to the evaluator to catch this.

## Round 6 — TrOCR with aligned GT
The core flaw in Round 5 was that Tesseract's garbled reading of degraded patches was used as the TrOCR supervision target. The model learned to generate patterns that make TrOCR output the same garbage — not to actually enhance the image.

Fix: align the SmartDoc document-level ground truth to each patch at preprocessing time (sliding window word-overlap search), save as `aligned_gt.txt`, and use that as the TrOCR target instead. Tesseract's reading continues to be used only as the FiLM conditioning signal (a guide, not a target).

Results were poor. Val loss barely moved (11.66 → 11.02 over 10 epochs). Both problem images were still flagged NOISE. The gradient signal from TrOCR was too weak: for patches that are too degraded to read, TrOCR can't process them at all, so gradients don't flow back meaningfully.

## Round 7 — SmartDoc paired L1 (synthetic degradation)
Switched away from the TrOCR loss entirely. Instead: take real SmartDoc phone photos, apply synthetic degradation (gaussian blur, motion blur, JPEG compression, noise), and train with a direct pixel-level loss (MultiScaleL1 + 0.2 × GradientLoss) to recover the original photo.

This gives a clean, strong gradient signal on every sample regardless of whether the image is readable. 1,000 SmartDoc images × 4 patches × 4 degradation variants = 16,000 training pairs. Trained for 60 epochs. Val loss: 0.2299 → 0.1176 (plateau ~0.117 from epoch 30 onward).

### Round 7 results
Average effective miss rate delta: -2.4%. The two problem images:
- TestImage_1 (heavily blurred): 85.7% → 39.3% miss rate (-46.4%), word count preserved — no collapse
- TestImage_4: 30.6% → 38.5% (+7.9%), word count preserved and enhanced output is visually readable

Most importantly: the enhanced outputs are actual readable text, not grid patterns. The noise generation problem is gone. SmartDoc held-out WER evaluation showed +2.2% average delta, but most held-out images were skipped by the quality gate (they're clear enough already) — the relevant benchmark is the UsedImages test set.
