'''
    22.09.23. Seokju Lee
    Modified for python3
    Original code from https://github.com/Leo-LiHao/OpenCV-Python-Tutorials
'''
import cv2 as cv
import time
import numpy as np
from matplotlib import pyplot as plt
import pdb


figNum = 0

if __name__ == '__main__':
    Img = cv.imread('../../../Datas/lena.png')
    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(Img, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();

    T0 = time.time()
    ResizeImg = cv.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv.INTER_CUBIC)
    print('cv.INTER_CUBIC: ', time.time() - T0)
    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(ResizeImg, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();

    T0 = time.time()
    ResizeImg = cv.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv.INTER_LANCZOS4)
    print('cv.INTER_LANCZOS4: ', time.time() - T0)
    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(ResizeImg, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();

    T0 = time.time()
    ResizeImg = cv.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv.INTER_AREA)
    print('cv.INTER_AREA: ', time.time() - T0)
    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(ResizeImg, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();

    T0 = time.time()
    ResizeImg = cv.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv.INTER_LINEAR)
    print('cv.INTER_LINEAR: ', time.time() - T0)
    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(ResizeImg, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();

    T0 = time.time()
    ResizeImg = cv.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv.INTER_NEAREST)
    print('cv.INTER_NEAREST: ', time.time() - T0)
    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(ResizeImg, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();


    '''
        Q. Please discuss the mechanism of each interpolation. What are the pros and cons?
    '''

    pdb.set_trace()
