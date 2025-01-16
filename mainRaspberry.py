# main.py --> driver for the rest of the program
# November 14, 2024 - Iya - Created, set up threads
# Novemver 18, 2024 - Iya - Integrated other programs
# November 20, 2024 - Iya - Add reaction to button features
# November 21, 2024 - Iya - Changed to raspberry pi code
# November 24, 2024 - Iya - Add in dynamic image calls
# November 25, 2024 - Kaniesa, Hannah, Jasmehar, Iya - Integrate properly with hardware
# November 26, 2024 - Kaniesa, Hannah, Jasmehar, Iya - Integrate properly with hardware

import time
import threading
import imageToText
import converter
import os               # to check for new images
import shutil           # using to move image around
import RPi.GPIO as GPIO
import MotorCalls

playing = True              # paused or not
showTime = 1                # how long displayed
outputArrays = []           # array of arrays of piston commands
lock = threading.Lock()     # ensure threads don't access same data at once
curI = 0                    # outputArray index
tolerance = 4               # How much error we accept from image to text (1/tolerance * 100%)
threadList = []             # list of running threads (these reset with new image)
goThread = True             # Used to stop thread

btnPausePin = 23   
btnUpPin = 24
btnSlowPin = 12 
btnBackPin = 16   

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup buttons
GPIO.setup(btnPausePin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(btnUpPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(btnSlowPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(btnBackPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Get the instructions for the motors
def getOutArray(char):
    global outputArrays
    outArray = converter.get_braille_dots(char)

    with lock:  # Lock access to the shared motor_data array
        if outArray[7] == 1:
            preArray = [0, 0, 1, 1, 1, 1]
            outputArrays.append(preArray)
        elif outArray[6] == 1:
            preArray = [0, 0, 0, 0, 0, 1]
            outputArrays.append(preArray)

        preArray = outArray[0:6]
        outputArrays.append(preArray)

# Call the outputs function looping through instructions
def runOutputThread():
    # print("Process Results")
    global playing, showTime, curI
    while goThread:
        if not playing:  # if paused
            time.sleep(0.1)
            continue
        
        if curI < len(outputArrays): # in the instructions array
            with lock:  # Lock access to array while getting data
                output = outputArrays[curI]

            MotorCalls.callMotors(output)
            time.sleep(showTime)
            MotorCalls.resetMotors()

            with lock:  # Lock access to increment
                curI += 1

# From input text call set instruction function
def runInputThread(text):
    global goThread
    for char in text:
        getOutArray(char)
        if not goThread: # End the loop to kill the thread
            break

# Directly process the text in a for loop
def startThreads(text):    

    # Output Thread
    outThread = threading.Thread(target=runOutputThread)
    outThread.daemon = True  # it exits with the program
    threadList.append(outThread)
    outThread.start()

    # Input Thread     
    inThread = threading.Thread(target=runInputThread, args=(text,))
    inThread.daemon = True
    threadList.append(inThread)
    inThread.start()

    inThread.join() # Let threads finish
    outThread.join()

# Begin processing image and start threads
def processImageEtEtc(address, tolerance):
    print("start threads", address)
    global threadList, outputArrays, playing, curI, goThread

    # Stop old threads (used if changing to new image inbetween)
    goThread = False
    for thread in threadList:
        if thread.is_alive():
            print("REstart")
            thread.join()  # Wait for the old thread to finish
    goThread = True

    print("Here")
    curI = 0
    outputArrays = []  # Clear instructs

    # Process the new image (outsourced to other programs)
    text = imageToText.handleImage(address, tolerance)
    print(text)
    if(text == "-1"): # Output error message
        text = "@@@"
    print(' '.join(converter.visual_braille_convert(text)))
    
    # main thread
    mainThread = threading.Thread(target=startThreads, args=(text,))
    mainThread.daemon = True  # stops with program ... prevents weird issues
    mainThread.start()
    threadList.append(mainThread)

    playing = False

# check for new images and make calls to restart
def checkForNewImage(pathToFolder):
    # print("new image")
    global playing
    fileSet = set() # of type set

    while True:
        time.sleep(1) 
        files = os.listdir(pathToFolder)
        
        new_images = [file for file in files if file.endswith('.jpg')] # new syntax called --> list comprehension [(add this)x for x in fruits if condition]

        for image in new_images:
            if image not in fileSet and not playing:
                fileSet.add(image)
                address = os.path.join(pathToFolder, image)

                # If paused restart with new image
                if not playing:
                    processImageEtEtc(address, tolerance)  
                    playing = False

# Button response (pause/unpause)
def buttonPause():
    global playing
    playing = not playing

# Button speed up 
def speedUp():
    global showTime
    if showTime > 0.05:  # Can't be zero
        showTime -= 0.05

# Button slow down
def slowDown():
    global showTime
    showTime += 0.05

# Button replay (go back 10 characters)
def replay():
    global curI
    with lock:  # Lock access to change place
        if curI >= 10:
            curI = curI - 10
        else: 
            curI = 0

# Move Images to another folder after program runs
# --> put used images in a different folder so when program restarts no confusion
def cleanup():
    # print("Clean up")

    if not os.path.exists("UsedImages"):
        os.makedirs("UsedImages") 
    
    # Move .jpg files
    for file in os.listdir("."):
        if file.endswith('.jpg'):
            shutil.move(os.path.join(".", file), os.path.join("UsedImages", file))

    # Raspberry Pi clean up
    MotorCalls.cleanup()
    GPIO.cleanup()

# Set up button pins
GPIO.setup(btnPausePin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  
GPIO.setup(btnUpPin, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
GPIO.setup(btnSlowPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(btnBackPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  

# button feedback setup      Notes: falling means look for button press, bouncetime ensures it only considers one press every 0.3 seconds
GPIO.add_event_detect(btnPausePin, GPIO.FALLING, callback=buttonPause, bouncetime=300)
GPIO.add_event_detect(btnUpPin, GPIO.FALLING, callback=speedUp, bouncetime=300)
GPIO.add_event_detect(btnSlowPin, GPIO.FALLING, callback=slowDown, bouncetime=300)
GPIO.add_event_detect(btnBackPin, GPIO.FALLING, callback=replay, bouncetime=300)

# Main code
locationThread = "."  
imageCheckerThread = threading.Thread(target=checkForNewImage, args=(locationThread,))
imageCheckerThread.daemon = True
imageCheckerThread.start()
cleanup()
