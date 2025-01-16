# sharpenImage --> apply preprocessing to the image
# November 10, November - Iya - Attempt the techniques
# November 11, November - Iya - Try those things out

import numpy as np
import cv2
from PIL import Image, ImageOps

# grayscale, binarize, noise removal, deskewing to preprocess images
def preprocessing(imagePath):
    image = Image.open(imagePath)

    # Sometimes images rotated randomly, looking at metadata to correct
    try:
        image = ImageOps.exif_transpose(image)
    except (AttributeError, KeyError, IndexError): # in case no metadata provided
        pass

    image = image.convert('L') # Greyscale
    image = image.point(lambda p: 255 if p > 128 else 0) # Binarize (make black or white)

    # Noise reduction
    cvData = np.array(image)
    cvData = cv2.medianBlur(cvData, 3)

    # Deskewing the image (if the text is rotated)
    coords = np.column_stack(np.where(cvData > 0))  # Get the coordinates of non-zero pixels
    angle = cv2.minAreaRect(coords)[-1]  # Rectangle that incloses all the black-pixels
    
    # Correcting the angle if wrong
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle  # This is to rotate correctly in small angles

    # Correcting the orientation
    (h, w) = cvData.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotatedImage = cv2.warpAffine(cvData, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotatedImage

# Test Code
# RESULT OF TESTING ARE IN: TestResults\sharpenImageTesting.md
""" 
for i in range(1, 17):
    cv2.imwrite(f"TestResults\\PreprocessedImages\\processed_{i}.jpg", preprocessing(f"UsedImages\\TestImage_{i}.jpg"))
 """