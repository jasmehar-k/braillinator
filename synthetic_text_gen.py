import argparse
import os
import random
import textwrap

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = [
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/Palatino.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]

# Varied sample text blocks to simulate real printed documents
_PARAGRAPHS = [
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! The five boxing wizards jump quickly.",

    "In publishing and graphic design, Lorem ipsum is a placeholder text commonly used to "
    "demonstrate the visual form of a document or a typeface without relying on meaningful content.",

    "Optical character recognition (OCR) is the process of converting images of typed, handwritten "
    "or printed text into machine-encoded text. OCR is widely used for data entry from printed records.",

    "Machine learning is a method of data analysis that automates analytical model building. "
    "It is based on the idea that systems can learn from data, identify patterns and make decisions.",

    "The Python programming language emphasizes code readability, simplicity, and explicitness. "
    "Python supports multiple programming paradigms, including structured, object-oriented programming.",

    "Digital image processing involves using computer algorithms to perform image processing on digital images. "
    "As a subcategory or field of digital signal processing, digital image processing has advantages.",

    "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence "
    "concerned with the interactions between computers and human language.",

    "Tesseract is an optical character recognition engine developed by HP in the 1980s. "
    "In 2005 Tesseract was open sourced by HP. It is now developed by Google.",

    "A neural network is a series of algorithms that endeavors to recognize underlying relationships "
    "in a set of data through a process that mimics the way the human brain operates.",

    "Computer vision is an interdisciplinary scientific field that deals with how computers can gain "
    "high-level understanding from digital images or videos. It seeks to automate tasks that the human visual system can do.",

    "The Raspberry Pi is a low cost, credit-card sized computer that plugs into a computer monitor or TV, "
    "and uses a standard keyboard and mouse. It is capable of doing everything you'd expect a desktop computer to do.",

    "Braille is a tactile writing system used by people who are visually impaired. "
    "Refreshable braille displays allow blind computer users to read text on a computer.",

    "Signal processing is an electrical engineering subfield that focuses on analysing, modifying and "
    "synthesising signals such as sound, images and scientific measurements.",

    "Information theory is the scientific study of the quantification, storage, and communication of information. "
    "The field was fundamentally established by Claude Shannon's 1948 paper.",

    "Convolutional neural networks are regularized types of feed-forward neural networks that learn features "
    "via filter optimization. They have applications in image recognition, video analysis and natural language processing.",

    "The U-Net architecture was developed for biomedical image segmentation at the University of Freiburg. "
    "The network is based on fully convolutional networks and its architecture was modified to work with "
    "fewer training images and to yield more precise segmentations.",

    "Feature-wise Linear Modulation (FiLM) conditions neural networks on auxiliary information by "
    "applying an affine transformation to intermediate feature maps. It enables flexible conditioning.",

    "Residual learning reformulates the layers as learning residual functions with reference to the "
    "layer inputs, instead of learning unreferenced functions. Residual networks are easier to optimize.",

    "Data augmentation techniques artificially expand the training dataset using transformations like "
    "rotation, flipping, cropping, and color jitter. This helps prevent overfitting on small datasets.",

    "Transfer learning is a research problem in machine learning that focuses on storing knowledge "
    "gained while solving one problem and applying it to a different but related problem.",
]


def _build_page_text(num_paragraphs: int = 6) -> str:
    paras = random.sample(_PARAGRAPHS, min(num_paragraphs, len(_PARAGRAPHS)))
    random.shuffle(paras)
    return "\n\n".join(paras)


def render_page(font_path: str, font_size: int, text: str, out_path: str) -> None:
    img = Image.new("L", (1200, 1600), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    wrap_width = max(30, int(80 * 14 / font_size))
    x, y = 60, 60
    line_spacing = int(font_size * 1.5)

    for para in text.split("\n\n"):
        lines = textwrap.wrap(para, width=wrap_width)
        for line in lines:
            if y + line_spacing > 1560:
                break
            draw.text((x, y), line, fill=0, font=font)
            y += line_spacing
        y += line_spacing  # extra gap between paragraphs
        if y > 1560:
            break

    img.save(out_path)


def generate_synthetic_images(output_dir: str = "synthetic_images", count: int = 200) -> None:
    os.makedirs(output_dir, exist_ok=True)

    available_fonts = [p for p in FONT_PATHS if os.path.exists(p)]
    if not available_fonts:
        raise RuntimeError("No system fonts found. Ensure running on macOS with standard fonts.")

    print(f"Using {len(available_fonts)} fonts: {[os.path.basename(f) for f in available_fonts]}")

    for i in range(count):
        random.seed(i)
        font_path = random.choice(available_fonts)
        font_size = random.randint(12, 22)
        num_paragraphs = random.randint(4, 8)
        text = _build_page_text(num_paragraphs)

        out_path = os.path.join(output_dir, f"page_{i:04d}.png")
        render_page(font_path, font_size, text, out_path)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Generated {i+1}/{count}: {os.path.basename(out_path)}")

    print(f"\nDone. {count} synthetic images written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic printed-text training images.")
    parser.add_argument("--output", default="synthetic_images", help="Output folder (default: synthetic_images)")
    parser.add_argument("--count", type=int, default=200, help="Number of images to generate (default: 200)")
    args = parser.parse_args()
    generate_synthetic_images(args.output, args.count)
