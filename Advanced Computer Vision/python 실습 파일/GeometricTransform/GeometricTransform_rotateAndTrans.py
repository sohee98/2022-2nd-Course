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
        	output reverse affine transformation. [0][1],[1][0] / [0][2],[1][2] values are exchanged each.

        Q. Please try different rotations, and repeat the process.
           RotateMatrix10 = cv.getRotationMatrix2D(center=(Img.shape[1]/2, Img.shape[0]/2),angle=10,scale=1)
           RotateMatrix20 = cv.getRotationMatrix2D(center=(Img.shape[1]/2, Img.shape[0]/2),angle=20,scale=1)
           RotateMatrix30 = cv.getRotationMatrix2D(center=(Img.shape[1]/2, Img.shape[0]/2),angle=30,scale=1)

           RotImg10 = cv.warpAffine(Img, RotateMatrix10, (Img.shape[0], Img.shape[1]))
           RotImg20 = cv.warpAffine(Img, RotateMatrix20, (Img.shape[0], Img.shape[1]))
           figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(RotImg10, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();
           figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(RotImg20, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();

            (Pdb) print(RotateMatrix10)
                [[  0.98480775   0.17364818 -40.56471825]
                 [ -0.17364818   0.98480775  48.34314871]]
            (Pdb) print(RotateMatrix20) 
                [[  0.93969262   0.34202014 -72.11846761]
                 [ -0.34202014   0.93969262 102.99584577]]
            (Pdb) print(RotateMatrix30)
                [[  0.8660254    0.5        -93.70250337]
                 [ -0.5          0.8660254  162.29749663]]



    '''

    M = np.vstack((RotateMatrix, np.array([0., 0., 1.])))
    InvM = np.linalg.inv(M)

    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(RotImg, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();
    figNum+=1; plt.figure(figNum); plt.imshow(cv.cvtColor(RotImg2, cv.COLOR_BGR2RGB)); plt.ion(); plt.show();


    
    pdb.set_trace()

