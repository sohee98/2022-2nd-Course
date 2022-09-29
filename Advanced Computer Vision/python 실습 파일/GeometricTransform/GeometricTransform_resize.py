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
        	cv.INTER_CUBIC:  0.001699686050415039
			cv.INTER_LANCZOS4:  0.005689382553100586
			cv.INTER_AREA:  0.00046753883361816406
			cv.INTER_LINEAR:  0.0006177425384521484
			cv.INTER_NEAREST:  0.00051116943359375

			cv.INTER_CUBIC:  a bicubic interpolation over 4x4 pixel neighborhood - use 16 pixels. slower than LINEAR but better quality
			cv.INTER_LANCZOS4:  a Lanczos interpolation over 8x8 pixel neighborhood - use 64 pixels. more complicated, slow but high quality
			cv.INTER_AREA:  resampling using pixel area relation. preferred method for image decimation.  similar to NEAREST - fastest. 
			cv.INTER_LINEAR:  a bilinear interpolation (used by default) - fast, good quality
			cv.INTER_NEAREST:   a nearest-neighbor interpolation - fast but low quality 


    '''

    pdb.set_trace()
