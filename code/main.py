# TensorFlow and tf.keras
from pyexpat import model
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from cv2 import COLOR_RGB2GRAY
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import mediapipe as mp
import time
import os
new_model = tf.keras.models.load_model('saved_model/my_model')


# Check its architecture
# new_model.summary()
imagepaths = []
# Go through all the files and subdirectories inside a folder and save path to images inside list
for root, dirs, files in os.walk("./test_data", topdown=False): 
#   print(root,"hii", dirs,"he", files)
  for name in files:
    path = os.path.join(root, name)

    if path.endswith("jpg"): # We want only the images
      imagepaths.append(path)

print(len(imagepaths),imagepaths) # If > 0, then a PNG image was loaded

X = [] # Image data

# Loops through imagepaths to load images and labels into arrays
for path in imagepaths:
  img = cv2.imread(path) # Reads image and returns np.array
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
  img = cv2.resize(img, (320, 120)) # Reduce image size so training can be faster
  X.append(img)

# Turn X and y into np.array to speed up train_test_split
X = np.array(X, dtype="uint8")
X = X.reshape(len(imagepaths), 120, 320, 1) # Needed to reshape so CNN knows it's different images

# print(len(X),X)

predictions = new_model.predict(X) # Make predictions towards the test set
np.argmax(predictions[0]) # If same, got it right
for x in range(0,8):
  y=predictions[x]
  result=np.where(y==np.amax(y))
  print(x,"hiii",result[0]+1)



# Hand={0:"Right",1:"Left"}


# cam=cv2.VideoCapture(0)
# mpHands=mp.solutions.hands
# hands=mpHands.Hands(static_image_mode=False,
#                max_num_hands=1,
#                model_complexity=1,
#                min_detection_confidence=0.8,
#                min_tracking_confidence=0.8)
# mpDraw=mp.solutions.drawing_utils
# pTime = 0
# cTime = 0


# img=cv2.imread("test_data/first.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
# img = cv2.resize(img, (320, 120))


# predictions = new_model.predict(img) # Make predictions towards the test set
# while True:
#     cv2.imshow("Camera",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# while True:
#     ret, img=cam.read()
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
#     img = cv2.resize(img, (320, 120))
#     # results=hands.process(imgRGB)

#     # loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
#     # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

#     print(new_model.predict(img).shape)
#     # if results.multi_hand_landmarks:
#     #         for handLms in results.multi_hand_landmarks:
#     #             # print("hiii",handLms.landmark)
#     #             for id, lm in enumerate(handLms.landmark):
                    
#     #                 if id==4:
                        
#     #                     print(id,lm)

#     #                 h,w,c=img.shape
#     #                 cx,cy=int(lm.x*w),int(lm.y*h)
                    
#     #                 # cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 150, 183), 2)
                
#     #             mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

#     cv2.imshow("Camera",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


