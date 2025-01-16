import time
import threading
import imageToText
import converter
import os
import shutil
import tkinter as tk

# Constants and Global Variables
tolerance = 4
outputArrays = []
textArray = []
lock = threading.Lock()
playing = True
curI = 0
showTime = 0.5
goThread = True
threadList = []


# Motor Control Functions (Replace with actual calls)
def callMotors(pinArray, app, char, braille):
    print("\r",end="")
    global curI, showTime
    print("position: ",curI,"\ttime delay: \t",showTime, end='')
    print('\tMotor Output:\t', end='')
    for i in range(0, 6):
        if pinArray[i]:
            print('.', end='')
        else:
            print('_', end='')

    #print('Motor Output:', pinArray)
    app.updateBigCharDisplay(char, braille)

def resetMotors(app):
    print('\tReset Output:______', end='')  
    app.updateBigCharDisplay('', '')

def getOutArray(char):
    global outputArrays
    outArray = converter.get_braille_dots(char)
    if outArray:
        outputArrays.append(outArray[0:6])

def runOutputThread(app):
    global playing, showTime, curI, textArray
    while goThread:
        if not playing:
            time.sleep(0.1)
            continue
        if curI < len(outputArrays):
            with lock:
                output = outputArrays[curI]
            callMotors(output, app, textArray[curI], converter.get_braille_char(textArray[curI]))
            time.sleep(showTime)
            resetMotors(app)
            with lock:
                curI += 1

def runInputThread(text):
    global goThread
    for char in text:
        getOutArray(char)
        if not goThread:
            break

def startThreads(text, app):
    outThread = threading.Thread(target=runOutputThread, args=(app,))
    inThread = threading.Thread(target=runInputThread, args=(text,))
    outThread.daemon = True
    inThread.daemon = True
    outThread.start()
    inThread.start()
    inThread.join()
    outThread.join()

def processImageEtEtc(address, tolerance, app):
    global threadList, outputArrays, playing, curI, goThread, textArray
    print("Start processing image:", address)

    goThread = False
    for thread in threadList:
        if thread.is_alive():
            thread.join()

    goThread = True
    curI = 0
    outputArrays = []
    
    text = imageToText.handleImage(address, tolerance).strip()
    if text:
        textArray = list(text)
        print("\n")
        print('Text Array:', textArray)
        print("\n\n")
    else:
        print("Error: No valid text.")

    mainThread = threading.Thread(target=startThreads, args=(text, app))
    mainThread.daemon = True
    mainThread.start()
    threadList.append(mainThread)

    playing = False

def checkForNewImage(pathToFolder, app):
    global playing  # To ensure it checks the global state of "playing"
    fileSet = set()  # To track files that have already been processed

    while True:
        time.sleep(1)
        files = os.listdir(pathToFolder)

        new_images = [file for file in files if file.endswith('.jpg')]

        for image in new_images:
            if image not in fileSet and not playing:
                fileSet.add(image)
                address = os.path.join(pathToFolder, image)

                # If paused, restart with new image
                if not playing:
                    processImageEtEtc(address, tolerance, app)

                playing = False

# Button Functions
def buttonPause():
    global playing
    playing = not playing

def speedUp():
    global showTime
    if showTime > 0.5:  # Can't be zero
        showTime -= 0.5

def slowDown():
    global showTime
    showTime += 0.5

def replay():
    global curI
    with lock:  # Lock access to change place
        if curI >= 10:
            curI = curI - 10
        else:
            curI = -1

# Move Images to another folder after program runs
def cleanup():
    if not os.path.exists("UsedImages"):
        os.makedirs("UsedImages") 
    
    # Move .jpg files
    for file in os.listdir("."):
        if file.endswith('.jpg'):
            shutil.move(os.path.join(".", file), os.path.join("UsedImages", file))

    print("Images moved succesfully")
    print("\n\n\n\n\n")

# GUI Class
class BrailleApp:
    def __init__(self, root):
        self.root = root
        self.setupGUI()

    def setupGUI(self):
        self.root.title("Braille Display Control")
        window_width = 500
        window_height = 500
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        position_top = 100
        position_left = 1100

        self.root.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")
        self.root.attributes('-topmost', True)

        primary_color = "#0b3c49" 
        bg_color = "#FFFFFF"       
        button_color = "#0b3c49"
        button_text_color = "#FFFFFF"  

        frameButtons = tk.Frame(self.root, bg=bg_color)
        frameButtons.pack(pady=20)

        btnPause = tk.Button(frameButtons, text="Play/Pause", command=buttonPause, font=("Helvetica", 14, "bold"), bg=button_color, fg=button_text_color, width=12, height=2, relief="flat")
        btnPause.pack(pady=5)

        btnSpeedUp = tk.Button(frameButtons, text="Speed Up", command=speedUp, font=("Helvetica", 14, "bold"), bg=button_color, fg=button_text_color, width=12, height=2, relief="flat")
        btnSpeedUp.pack(pady=5)

        btnSlowDown = tk.Button(frameButtons, text="Slow Down", command=slowDown, font=("Helvetica", 14, "bold"), bg=button_color, fg=button_text_color, width=12, height=2, relief="flat")
        btnSlowDown.pack(pady=5)

        btnBack = tk.Button(frameButtons, text="Back", command=replay, font=("Helvetica", 14, "bold"), bg=button_color, fg=button_text_color, width=12, height=2, relief="flat")
        btnBack.pack(pady=5)

        frameChars = tk.Frame(self.root)
        frameChars.pack(pady=20)

        self.bigCharLabel1 = tk.Label(frameChars, text=" ", font=("Helvetica", 36, "bold"), width=6, height=2, bg="lightgray", relief="solid")
        self.bigCharLabel1.grid(row=0, column=0, padx=10)

        self.bigCharLabel2 = tk.Label(frameChars, text=" ", font=("Helvetica", 36, "bold"), width=6, height=2, bg="lightgray", relief="solid")
        self.bigCharLabel2.grid(row=0, column=1, padx=10)

        frameTextBoxes = tk.Frame(self.root)
        frameTextBoxes.pack(pady=20)

        self.textBox1 = tk.Entry(frameTextBoxes, font=("Helvetica", 24, "bold"), bg="white", fg="black", bd=2, relief="solid", width=6)
        self.textBox1.grid(row=0, column=0, padx=10)

        self.textBox2 = tk.Entry(frameTextBoxes, font=("Helvetica", 24, "bold"), bg="white", fg="black", bd=2, relief="solid", width=6)
        self.textBox2.grid(row=0, column=1, padx=10)

        locationThread = "."  
        imageCheckerThread = threading.Thread(target=checkForNewImage, args=(locationThread, self))
        imageCheckerThread.daemon = True
        imageCheckerThread.start()

    def updateBigCharDisplay(self, char, braille):
        self.bigCharLabel1.config(text=char)
        self.bigCharLabel2.config(text=braille)

        self.textBox1.delete(0, tk.END)  # Clear current text
        self.textBox1.insert(tk.END, char)
        self.textBox2.delete(0, tk.END)  # Clear current text
        self.textBox2.insert(tk.END, braille)

# Main function to start the GUI
def main():
    print("\n\n\n\n\n")
    root = tk.Tk()
    app = BrailleApp(root)
    root.mainloop()
    cleanup()  # Ensure cleanup when the program exits

main()
