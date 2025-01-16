# normalize.py --> Regex to format text read from pytesseract
# November 5, 2024 - Iya - Created
# November 12 2024 - Iya - Updated RegEx to handle commas
import re

def normalize_newlines(text):
    # Regex that finds random line breaks between a sentence and removes them
    reorg = re.sub(r'(\w|,)\s*[\n]+(\w)', r'\1 \2', text)
    return reorg


# RESULT OF TESTING ARE IN: TestResults\normalizeTesting.txt
""" 
# Test cases
testCaseDictionary = [
    {
        'input': "This is how it,\nread the \ntext sometimes",
        'expected': "This is how it, read the text sometimes"
    },
    {
        'input': "This is how it\nworks out sometimes",
        'expected': "This is how it works out sometimes"
    },
    {
        'input': "Hello there,\nthis is a test",
        'expected': "Hello there, this is a test"
    },
    {
        'input': "Multiple\n\n\nnewlines\n\nare here",
        'expected': "Multiple newlines are here"
    },
    {
        'input': "Nothing to change.",
        'expected': "Nothing to change."
    }
]

for i, testCase in enumerate(testCaseDictionary, start=1):
    input = testCase['input']
    expectedOutput = testCase['expected']
    result = normalize_newlines(input)
    
    print(f"Test {i}:")
    print(f"Original: {input}")
    print(f"Normalized: {result}")
    print(f"Expected: {expectedOutput}")
    print("Pass" if result == expectedOutput else "Fail")
    print("------------------------------------------\n\n") 
"""