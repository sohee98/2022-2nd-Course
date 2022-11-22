'''
    2022.11.15. Seokju Lee @ EE7107
    Please download Haar Cascades from:
    https://github.com/opencv/opencv/tree/master/data/haarcascades

'''

import numpy as np
import cv2
import pdb

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
upperbody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
lowerbody_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

# img = cv2.imread('image_1.jpg')  	# 1
# img = cv2.imread('image_2.jpg')		# many people
# img = cv2.imread('image_3.jpg')		# crowd
# img = cv2.imread('image_4.jpg')		# people + body
img = cv2.imread('image_5.jpeg')		# body
# img = cv2.imread('image_6.jpeg')		# body
# img = cv2.imread('image_7.png')		# body
# img = cv2.imread('images.png')		# body
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5) 		# faces : [(x, y, w, h), ... ]
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    smiles = smile_cascade.detectMultiScale(roi_gray)
    for (sx,sy,sw,sh) in smiles:
        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)

fullbody = fullbody_cascade.detectMultiScale(gray, 1.3, 5)
# pdb.set_trace()
for (sx,sy,sw,sh) in fullbody:
    img = cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(255,0,153),2)
upperbody = upperbody_cascade.detectMultiScale(gray, 1.3, 5)
for (sx,sy,sw,sh) in upperbody:
    img = cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(0,102,255),2)
lowerbody = lowerbody_cascade.detectMultiScale(gray, 1.3, 5)
for (sx,sy,sw,sh) in lowerbody:
    img = cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(102,51,255),2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

pdb.set_trace()

'''
(Pdb) faces
array([[ 99, 118,  57,  57],
       [497,  89,  58,  58],
       [545, 171,  64,  64],
       [277,  44,  56,  56],
       [432, 242,  66,  66],
       [624, 110,  67,  67],
       [ 62,  25,  55,  55],
       [420,   7,  55,  55],
       [558,  26,  55,  55],
       [381,  66,  59,  59],
       [356, 165,  67,  67],
       [258, 222,  82,  82],
       [591, 284,  78,  78],
       [179, 118,  79,  79],
       [493, 362,  76,  76],
       [653, 416, 100, 100],
       [156, 436,  87,  87],
       [335, 418,  87,  87]], dtype=int32)

(Pdb) eyes
array([[16, 23, 22, 22],
       [44, 21, 24, 24]], dtype=int32)

(Pdb) roi_gray.shape
(87, 87)

(Pdb) roi_color.shape
(87, 87, 3)

'''