# converter.py --> Read from Braille.json and translate back and forth between braille
# November 12, 2024 - Iya - Created Basic translation **Must still handle number and capital conversion correctly
# November 18, 2024 - Iya - Added in capitals and number handling

import json

with open('braille.json', 'r', encoding='utf-8') as f: #Note 'r' = readMode, 'utf-8' = Handles all unicode 
    braille_data = json.load(f) # Get JSON contents

# Convert to array used for piston output
def braille_to_binary_array(dots):
    binary_array =   [0, 0, 0, 0, 0, 0, 0, 0]
    # binary_array = [a, b, c, d, e, f, g, h]
    #   a   d
    #   b   e
    #   c   f
    # g = capital letter so ⠠ should preceed the letter
    # h = number so ⠼ should preceed the number

    for dot in dots:
        if dot == -1:
            binary_array[6] = 1
        elif dot == -2:
            binary_array[7] = 1
        else:
            binary_array[dot - 1] = 1
    
    return binary_array

# Get the array corresponding to the braille from the english characters
def get_braille_dots(character):
    if character in braille_data['braille']['letters']:
        dots = braille_data['braille']['letters'][character]['dots']
    
    elif character in braille_data['braille']['uppercase_letters']:
        dots = braille_data['braille']['uppercase_letters'][character]['dots']
    
    elif character in braille_data['braille']['digits']:
        dots = braille_data['braille']['digits'][character]['dots']
    
    elif character in braille_data['braille']['punctuation']:
        dots = braille_data['braille']['punctuation'][character]['dots']
    
    elif character == ' ':
        dots = [0,0,0,0,0,0,0,0]
    
    else:
        return None
    return braille_to_binary_array(dots)

# Get the ⠃⠗⠁⠊⠇⠇⠑ from the english characters
def get_braille_char(character):
    if character in braille_data['braille']['letters']:
        return braille_data['braille']['letters'][character]['braille']
    
    elif character in braille_data['braille']['uppercase_letters']:
        return braille_data['braille']['uppercase_letters'][character]['braille']
    
    elif character in braille_data['braille']['digits']:
        return braille_data['braille']['digits'][character]['braille']
    
    elif character in braille_data['braille']['punctuation']:
        return braille_data['braille']['punctuation'][character]['braille']
    elif character == ' ':
        return ' '
    else:
        return ''

# Get the english characters from the ⠃⠗⠁⠊⠇⠇⠑ 
def get_eng_char(character): 
    if character == ' ':
        return character
    for letter, braille_info in braille_data['braille']['letters'].items():
        if braille_info['braille'] == character:
            return letter
    for letter, braille_info in braille_data['braille']['uppercase_letters'].items():
        if braille_info['braille'] == character:
            return letter
    for letter, braille_info in braille_data['braille']['digits'].items():
        if braille_info['braille'] == character:
            return letter
    for letter, braille_info in braille_data['braille']['punctuation'].items():
        if braille_info['braille'] == character:
            return letter
    return '!' 

# Convert an entire string from English to Braille
def visual_braille_convert(strings):
    converted = ''
    for character in strings:
        converted += get_braille_char(character)
    return converted

# Convert an entire string from Braille to English
def braille_to_text(strings):
    converted = ''
    temp = ''

    for character in strings:
        if character == '⠼' or character == '⠠':
            temp = character
        else:
            converted += get_eng_char(temp + character)
            temp = ''
    return converted


# Test Code
# RESULT OF TESTING ARE IN: TestResults\converterTesting.txt
""" 
testCases = {
    'visual_braille_convert': {
        "Test 1": {
            'input': "hello", 
            'expected_output': "⠓⠑⠇⠇⠕"
        },
        "Test 2": {
            'input': "braille", 
            'expected_output': "⠃⠗⠁⠊⠇⠇⠑"
        },
        "Test 3": {
            'input': "world", 
            'expected_output': "⠺⠕⠗⠇⠙"
        },
        "Test 4": {
            'input': "upper", 
            'expected_output': "⠥⠏⠏⠑⠗"
        },
        "Test 5": {
            'input': "space test", 
            'expected_output': "⠎⠏⠁⠉⠑ ⠞⠑⠎⠞"
        },
        "Test 6": {
            'input': "hello, world!",
            'expected_output': "⠓⠑⠇⠇⠕⠂ ⠺⠕⠗⠇⠙⠖"
        },
        "Test 7": {
            'input': "Hello",
            'expected_output': "⠠⠓⠑⠇⠇⠕"
        },
        "Test 8": {
            'input': "@@@",
            'expected_output': "⠿⠿⠿"
        }
    },
    'braille_to_text': {
        "Test 1": {
            'input': "⠓⠑⠇⠇⠕", 
            'expected_output': "hello"
        },
        "Test 2": {
            'input': "⠃⠗⠁⠊⠇⠇⠑", 
            'expected_output': "braille"
        },
        "Test 3": {
            'input': "⠺⠕⠗⠇⠙", 
            'expected_output': "world"
        },
        "Test 4": {
            'input': "⠥⠏⠏⠑⠗", 
            'expected_output': "upper"
        },
        "Test 5": {
            'input': "⠎⠏⠁⠉⠑ ⠞⠑⠎⠞", 
            'expected_output': "space test"
        },
        "Test 6": {
            'input': "⠓⠑⠇⠇⠕⠂ ⠺⠕⠗⠇⠙⠖", 
            'expected_output': "hello, world!"
        },
        "Test 7": {
            'input': "⠠⠓⠑⠇⠇⠕", 
            'expected_output': "Hello"
        },
        "Test 8": {
            'input': "⠠⠥⠏⠏⠑⠗", 
            'expected_output': "Upper"
        }
    },
    'braille_to_binary_array': {
        "Test 1": {
            'input': [1, 2],
            'expected_output': [1, 1, 0, 0, 0, 0, 0, 0]
        },
        "Test 2": {
            'input': [1, 3, 4],
            'expected_output': [1, 0, 1, 1, 0, 0, 0, 0]
        },
        "Test 3": {
            'input': [1, 4, -1],
            'expected_output': [1, 0, 0, 1, 0, 0, 1, 0]
        },
        "Test 4": {
            'input': [2, 5, 6],
            'expected_output': [0, 1, 0, 0, 1, 1, 0, 0]
        },
        "Test 5": {
            'input': [1, 2, 3, 4],
            'expected_output': [1, 1, 1, 1, 0, 0, 0, 0]
        },
        "Test 6": {
            'input': [-2, 3, 5],
            'expected_output': [0, 0, 1, 0, 1, 0, 0, 1]
        }
    }
}

for function, cases in testCases.items():
    print(f"\nTesting {function}")
    for test_case, data in cases.items():
        input = data['input']
        expectd = data['expected_output']
        
        # Call the appropriate function
        if function == 'visual_braille_convert':
            output = visual_braille_convert(input)
        elif function == 'braille_to_text':
            output = braille_to_text(input)
        elif function == 'braille_to_binary_array':
            output = braille_to_binary_array(input)
      
        if output == expectd:
            print(f"{test_case}: Passed \t Input: {input} Output: {expectd}")
        else:
            print(f"{test_case}: FAILED \t Input: {input} Expected: {expectd} Actual: {output}")

 """