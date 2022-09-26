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
    AffineMatrix = np.array([[1, 0, 100],
                             [0, 1,  50]], dtype=np.float32)
    DstImg = cv.warpAffine(Img, AffineMatrix, (Img.shape[0]+200, Img.shape[1]+200), borderValue=(155, 155, 155))

    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(Img, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();
    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(DstImg, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();
    '''
        Q. Why do we need cv.cvtColor? originally BGR => convert to RGB
    '''
    # Img.shape = (512, 512, 3)
    RotateMatrix = cv.getRotationMatrix2D(center=(Img.shape[1]/2, Img.shape[0]/2),
                                           angle=90,
                                           scale=1)
    print('(Img.shape[1]/2, Img.shape[0]/2):', (Img.shape[1]/2, Img.shape[0]/2))
    print('Rotate Matrix:', RotateMatrix)
    '''
        (Img.shape[1]/2, Img.shape[0]/2): (256.0, 256.0)
		Rotate Matrix: [[ 6.12323400e-17  1.00000000e+00 -2.84217094e-14]
						[-1.00000000e+00  6.12323400e-17  5.12000000e+02]]
        Q. Please discuss how to generate rotation matrix.
        	[[cos(90) -sin(90) 0][sin(90) cos(90) 0][0 0 1]] => affine matrix 


    '''

    RotImg = cv.warpAffine(Img, RotateMatrix, (Img.shape[0], Img.shape[1]))

    CVInv_M = cv.invertAffineTransform(RotateMatrix)
    RotImg2 = cv.warpAffine(Img, CVInv_M, (Img.shape[0], Img.shape[1]))
    '''
    CVInv_M : array([[ 6.12323400e-17, -1.00000000e+00,  5.12000000e+02],
       				 [ 1.00000000e+00,  6.12323400e-17, -2.92924863e-15]])

        Q. Please discuss the meaning of inverse matrix.
        	output reverse affine transformation. third column 2 values are changed.

        Q. Please try different rotations, and repeat the process.
           RotateMatrix2 = cv.getRotationMatrix2D(center=(Img.shape[1]/2, Img.shape[0]/2),angle=180,scale=1)
           RotImg3 = cv.warpAffine(Img, RotateMatrix2, (Img.shape[0], Img.shape[1]))
           figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(RotImg3, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();

    '''

    M = np.vstack((RotateMatrix, np.array([0., 0., 1.])))
    InvM = np.linalg.inv(M)

    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(RotImg, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();
    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(RotImg2, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();


    
    pdb.set_trace()

