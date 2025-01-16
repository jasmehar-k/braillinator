# autoCorrect.py --> autocorrects text counting frequency of words to avoid autocorrecting names and checks proportion of mispelled words
# November 2, 2024 - Iya - Created
# November 10, 2024 - Iya - Added frequency check 

from spellchecker import SpellChecker   # Dictionary and spellcheck
import Levenshtein                      # Check difference between words using Levenshtein Distance (number of changes to get from one to the other)
import re 


def autoCorrect(text):
    spellchecker = SpellChecker()
    words = re.findall(r'\w+|[^\w\s]', text)  #match words and punctuation separately
    misspelled = spellchecker.unknown(words)

    # Track if misspelled word appears multiple times (could be a name)
    for word in misspelled:
        if text.lower().count(word) >= 3:  # must lowercase the input text as spellchecker makes everything in misspelled lowercase
            spellchecker.word_frequency.add(word)

    correctedWords = []

    for word in words:
        if word.isalpha():  # only if a word and not punctuation
            correctWord = spellchecker.correction(word)

            # Only autocorrect if it is a small difference
            if correctWord is not None and Levenshtein.distance(word, correctWord) <= (len(word) / 2): 
                correctedWords.append(correctWord)
            else:
                correctedWords.append(word)
        else: # case of punctuation
            correctedWords.append(word)

    formattedWords = []

    for word in correctedWords:
        if word.isalnum():  # alphanumeric
            formattedWords.append(f" {word}")  # put space before it
        else:
            formattedWords.append(word) 

    result = ''.join(formattedWords) # concatentate everything
    return result.strip() # remove any extra spaces and send

# Checks if more than 1/toleranceNumber of words in the text are misspelled
def misspelledCount(text, toleranceNumber):
    spellchecker = SpellChecker()
    words = re.findall(r'\w+', text) # Need this to ignore punctuation
    misspelled = spellchecker.unknown(words)
    mistakes = len(misspelled)
    allWords = len(text.split())

    #Debugging Purposes
    #print("The error rates",float(mistakes/allWords)," - ", float(1/toleranceNumber))
    #print("The mispelled words are: ", misspelled)
    
    if (float(mistakes/allWords) < float(1/toleranceNumber)):
        return 1
    else:
        return 0

# Test Code
# RESULT OF TESTING ARE IN: TestResults\autoCorrectTesting.txt
""" testCaseDictionary = [
    {
        'input': "Today she ran to the stoore.",
        'expected': {
            'misspelledCount': True,
            'autoCorrect': "Today she ran to the store."
        }
    },
    {
        'input': "Today she ran to the stoore.",
        'expected': {
            'misspelledCount': True,
            'autoCorrect': "Today she ran to the store."
        }
    },
    {
        'input': "Today alice ran to the stoore. alice bought aples and banannas. alice relly likes fuits.",
        'expected': {
            'misspelledCount': False,
            'autoCorrect': "Today alice ran to the store. alice bought apples and bananas. alice really likes .uits."
        }
    },
    {
        'input': "Today Alice ran to the store.",
        'expected': {
            'misspelledCount': True,
            'autoCorrect': "Today .* ran to the store."
        }
    },
    {
        'input': "The quik bwn fox jumpped ovr the lazzy dog.",
        'expected': {
            'misspelledCount': False,
            'autoCorrect': "The .* fox .* the .* dog."
        }
    }
]

for i, testCase in enumerate(testCaseDictionary, start=1):
    inText = testCase['input']
    expectedCorrectness = testCase['expected']['misspelledCount']
    expectedCorrected = testCase['expected']['autoCorrect']
    
    # run code
    correctness = misspelledCount(inText, 4) 
    outText = autoCorrect(inText)
    
    #check with regex
    corrected_match = bool(re.search(expectedCorrected, outText))

    # Print results
    print(f"Test {i}:")
    print(f"Input: {inText}")
    print(f"Correctness: Expected {expectedCorrectness} and actually {correctness}\n\t-->{(correctness == expectedCorrectness)}")
    print(f"AUTOCORRECT:\nExpected: {expectedCorrected} \nActual:   {outText}\n\t-->{corrected_match}")
    print("------------------------------------------\n\n")
 """