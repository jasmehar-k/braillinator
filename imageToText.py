# imageToText --> Convert an image to text
# October 28, 2024 - Iya - Created Example
# November 11, November - Iya - Optimize flow

import platform
import numpy as np
import pytesseract
from PIL import Image, ImageOps
import sharpenImage
import autoCorrect
import enhance
from normalize import normalize_newlines

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Take image address and tolerance for accuracy and convert to text
def handleImage(address, fractionTolerance):
    # Standard preprocessing + first OCR pass
    preprocessed = sharpenImage.preprocessing(address)
    initial_text = pytesseract.image_to_string(preprocessed, lang='eng') or ""

    # Quality gate: only run the U-Net when the initial OCR result is poor.
    # Requires at least 10 words so the misspelling rate is meaningful — very
    # short OCR results (nearly blank images) should not trigger enhancement.
    enough_words = len(initial_text.split()) >= 10
    if enough_words and not autoCorrect.misspelledCount(initial_text, fractionTolerance):
        # Initial OCR failed the quality check — attempt U-Net enhancement.
        # Load raw grayscale so the model can restore soft photo detail before
        # binarization, where blurry images lose the most information.
        pil = Image.open(address)
        try:
            pil = ImageOps.exif_transpose(pil)
        except Exception:
            pass
        raw = np.array(pil.convert("L"))
        enhanced_raw = enhance.enhance_image(raw, initial_text)
        image = sharpenImage.preprocess_array(enhanced_raw)
    else:
        # Initial OCR is already good enough — skip enhancement.
        image = preprocessed

    text = pytesseract.image_to_string(image, lang='eng')

    if text is None: # This will lead to output error code
        print("No text found")
        return ("-1")

    text = normalize_newlines(text)
    text = autoCorrect.autoCorrect(text)

    # Return text only if decent portion is properly spelled
    if (autoCorrect.misspelledCount(text, fractionTolerance)):
        text = normalize_newlines(text)
        return (text)
    else:
        print("Too Inaccurate")
        return ("-1")

# Test Code
# RESULT OF TESTING ARE IN: TestResults\imageToTextTesting.md
""" for i in range(1, 18):
    print(f"Image {i}\n")
    print(handleImage(f"UsedImages\\TestImage_{i}.jpg", 4))
    print("_______________________________\n\n")
 """
