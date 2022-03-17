import cv2 as cv
import numpy as np
import time
import pyautogui

detection=1
def controls(id):
    global detection
    # if id==8:
    #     if detection:
    #         detection=0 
    #     else:
    #         detection=1
    # print("ppppppppppppp",id)
    if detection:
        if id==0:
            pyautogui.hotkey('ctrl','s')
            time.sleep(0.2)
        elif id==1:
            pyautogui.hotkey('win','prtsc')
            time.sleep(0.2)
        elif id==2:
            pyautogui.hotkey('up')
        elif id==3:
            pyautogui.hotkey('down')
        elif id==5:
            pyautogui.hotkey('esc')
            time.sleep(0.2)
        elif id==8:
            pyautogui.hotkey('enter')
            time.sleep(0.2)
