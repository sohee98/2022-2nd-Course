'''
    22.10.11. Seokju Lee
    Modified for python3
    Original code from https://github.com/Leo-LiHao/OpenCV-Python-Tutorials
'''


import sys
sys.path.append('../../../')

import cv2
import time
import numpy as np
import Src.ToolBox.ImageProcessTool as IPT
from matplotlib import pyplot as plt
import pdb


RED   = (0, 0, 255)
BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)

def drawCircleBoard(imgSize, boardSize, circleDis, circleRadius, originPt, backgroundColor, foregroundColor):
    Canvas = np.zeros(imgSize, dtype=np.uint8)
    Canvas[:, :] = backgroundColor
    RowNum, ColNum = boardSize
    OriginPoint_2x1 = np.array(originPt).reshape(2, 1)
    Circles_2xn = np.array([0, 0]).reshape(2, 1)
    for row in range(RowNum):
        for col in range(ColNum):
            x = col * circleDis
            y = row * circleDis
            Point = np.array([[x], [y]])
            Circles_2xn = np.hstack((Circles_2xn, Point))
    Circles_2xn = Circles_2xn[:, 1:] + OriginPoint_2x1
    IPT.drawPoints(img=Canvas, pts_2xn=Circles_2xn, color=foregroundColor, radius=circleRadius)
    return Canvas


if __name__ == '__main__':
    # CircleImg = \
    #     drawCircleBoard(imgSize=(500, 500, 3), boardSize=(7, 7), circleDis=50,
    #                     circleRadius=int(50*0.3), originPt=(50, 50), backgroundColor=(0, 0, 0),
    #                     foregroundColor=(255, 255, 255))
    # CircleImg = \
    #     drawCircleBoard(imgSize=(500, 500), boardSize=(7, 7), circleDis=50,
    #                     circleRadius=int(50*0.3), originPt=(50, 50), backgroundColor=0,
    #                     foregroundColor=255)
    # SrcImg = CircleImg

    # FilePath = '../../../Datas/Chessboard.jpg'
    # FilePath = 'butterfly.jpg'
    # FilePath = 'flower.jpg'
    FilePath = 'dog_1.png'
    SrcImg = cv2.cvtColor(cv2.imread(FilePath), code=cv2.COLOR_BGR2RGB)

    GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_RGB2GRAY)
    # pdb.set_trace()

    ### Define SIFT ###
    sift = cv2.SIFT_create()
    
    ### Compute SIFT ###
    keypoints, descriptor = sift.detectAndCompute(GrayImg, None)
    print('keypoint:',len(keypoints), 'descriptor:', descriptor.shape)
    print(descriptor)

    ### Draw Keypoints ###
    img_draw = cv2.drawKeypoints(SrcImg, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    pdb.set_trace()

    ### Show Results ###
    plt.figure(1); plt.imshow(SrcImg); plt.colorbar(); plt.ion(); plt.show()
    plt.figure(2); plt.imshow(img_draw); plt.colorbar(); plt.ion(); plt.show()
    pdb.set_trace()


