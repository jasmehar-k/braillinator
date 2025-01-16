# imageToText --> Convert an image to text
# October 28, 2024 - Iya - Created Example
# November 11, November - Iya - Optimize flow

import pytesseract
import sharpenImage
import autoCorrect
from normalize import normalize_newlines

# Take image address and tolerance for accuracy and convert to text
def handleImage(address, fractionTolerance):
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    
    image = sharpenImage.preprocessing(address)
    text = pytesseract.image_to_string(image, lang='eng')

    if text is None: # This will lead to output error code
        print("No text found")
        return ("-1") 
    
    # print("You have gotten the text and checked if it none")
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