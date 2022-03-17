from atexit import register
from cProfile import label
from multiprocessing.sharedctypes import Value
from tkinter import *
from tkinter import messagebox
from tkinter.font import names
from tkinter.ttk import Labelframe
from PIL import Image, ImageTk
from tkinter import filedialog as fd
from matplotlib.pyplot import text
from time import sleep
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from assign import controls

from sqlite3 import converters
from zlib import Z_PARTIAL_FLUSH
import numpy as np
import face_recognition as fr
import os
import test_img
from Virtual_Mouse import mes


root = Tk()
root.title("segmentation_fault")
root.geometry('600x400')
root.configure(background="gray")
name = Label(root, text="name")
name.grid()
mylabel1 = Label(root, text="enter your name")
mylabel2 = Label(root, text="password")
var1 = StringVar()
var2 = StringVar()
mylabel1.grid(row=0, column=0, sticky=W, pady=2)
mylabel2.grid(row=1, column=0, sticky=W, pady=2)


def vclick():
    if var1.get() == "":
        messagebox.showwarning("warning", "fill the name field")
    elif var2.get() == "":
        messagebox.showwarning("warning", "fill the password field")

    
    top = Toplevel()
    button3 = Button(top, text='go for hand gestures',bd=5, command=handREC, padx=10)

    button3.grid(row=7, sticky=W, pady=5)

       # comment out kr dena neeche ki 2 line
    button4 = Button(top, text="go for face recognition(beta)", bd=5, command=frec, padx=10)
    button4.grid(row=9, sticky=W, pady=5)

    button7 = Button(top, text="enable mouse control", bd=5, command=mes, padx=10)

    button7.grid(row=11, sticky=W, pady=5)


    top.mainloop()


e1 = Entry(root, textvariable=var1)
e2 = Entry(root, show="*", textvariable=var2)








def clear():
    var1.set("")
    var2.set("")


button1 = Button(root, text='enter', bd=5, command=vclick)
button1.grid(row=3, column=0, pady=5)

button2 = Button(root, text='clear', bd=5, command=clear)
button2.grid(row=3, column=1, pady=5)




def frec():

    # print(os.listdir("code/"))
    # Images = []
    # path = "test_img"
    # names = []
    # myList = os.listdir(path)
    # print(myList)
    # for i in myList:
    #     print(f'hiiiiii
    # 
    # 
    # iiii{path}h/{i}')
    #     img = cv2.imread(f'{path}h/{i}')
    #     Images.append(img)
    #     print(Images)
    #     names.append(os.path.splitext(i)[0])
    # print(names)

    Images = []
    names=[]
    # Go through all the files and subdirectories inside a folder and save path to images inside list
    for root, dirs, files in os.walk("test_img", topdown=False): 
    #   print(root,"hii", dirs,"he", files)
        for name in files:
            path = os.path.join(root, name)

            if path.endswith("jpg"): # We want only the images
                Images.append(path)
                names.append(name)

        print(len(Images),Images,names) # If > 0, then a PNG image was loaded

    def findEncoding(Images):
        encodelist = []
        for item in Images:
            print(item)
            item=cv2.imread(item)
            img = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            encoding = fr.face_encodings(item)[0]
            encodelist.append(encoding)
        return encodelist

    encodeListKnown = findEncoding(Images)

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        facesAvilable = fr.face_locations(imgs)
        encodeFaces = fr.face_encodings(imgs, facesAvilable)

        for encodeFaces, faceLoc in zip(encodeFaces, facesAvilable):
            match = fr.compare_faces(encodeListKnown, encodeFaces)
            dis = fr.face_distance(encodeListKnown, encodeFaces)
            print(dis)
            matchindex = np.argmin(dis)

        if match[matchindex]:
            name = names[matchindex].lower()
            print(name)
            cv2.imshow('webcam', img)
        cv2.waitKey(1)


e1.grid(row=0, column=1, pady=2)
e2.grid(row=1, column=1, pady=2)
print(e1, e2)


def handREC():

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
    model = load_model('mp_hand_gesture')
    model.summary()
# Load class names
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)


# Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read each frame from the webcam
        _, frame = cap.read()

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # print(result)

        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                # mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]
                print("hiiii", prediction[0], classID)
                controls(classID)
                # sleep(1)
                # for x in range(0,8):
                #     y=prediction[x]
                #     result=np.where(y==np.amax(y))
                #     print(x,"hiii",result[0]+1)

        # # show the prediction on the frame
        # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the final output
        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break
# release the webcam and destroy all active windows

    cap.release()
    cv2.destroyAllWindows()


name = []
password = []


def regis():
    if var1.get() == "":
        messagebox.showwarning("warning", "fill the name field")
    elif var2.get() == "":
        messagebox.showwarning("warning", "fill the password field")

    name.append(var1.get())
    password.append(var2.get())
    messagebox.showinfo("registration", "you have successfully registered")
    print(name)
    print(password)
    with open("usr.txt",'a') as f:
       f.write(var1.get()+"\n")
    with open("pass.txt",'a') as g:
        g.write(var2.get()+"\n")


def rem():
    name.remove(var1.get())
    password.remove(var2.get())
    print(name)
    print(password)
    messagebox.showinfo("removal", "entry removed")
button5 = Button(root, text="go for registration",bd=5, command=regis, padx=10)
button5.grid(row=7, sticky=W, pady=5)
button6 = Button(root, text="delete registration",bd=5, command=rem, padx=10)
button6.grid(row=9, sticky=W, pady=5)
root.mainloop()

# name,email,password
