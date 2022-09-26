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

import sys
sys.path.append("../../../")
import Src.ToolBox.ImageProcessTool as IPT

'''
    Q. Please briefly discuss the contents of this code.

    Q. Please try different affine transformations, and discuss them.
'''


LeftPoint = [0, 0]
RightPoint = [0, 0]

def OnMouse1(event, x, y, flags, *args):
    global LeftPoint

    if cv.EVENT_LBUTTONDOWN == event:
        LeftPoint = [x, y]
    else:
        return

def OnMouse2(event, x, y, flags, *args):
    global RightPoint
    if cv.EVENT_RBUTTONDOWN == event:
        RightPoint = [x, y]
    else:
        return


if __name__ == '__main__':
    First = True
    SrcImg = cv.imread('../../../Datas/Paper3.jpg')
    SrcCanvas = np.zeros(SrcImg.shape, dtype=np.uint8)
    cv.namedWindow("Src", cv.WINDOW_NORMAL)
    # cv.namedWindow("Src")
    cv.setMouseCallback("Src", OnMouse1)
    # cv.namedWindow("Canvas")
    cv.namedWindow("Canvas", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Canvas", OnMouse2)

    SrcPoints = np.float32([[ 515.,  357.],
                            [ 708.,  365.],
                            [ 508.,  555.],
                            [ 736.,  562.]])
    CanvasPoints = np.float32([[0,0],[300,0],[0,300],[300,300]])
    while True:
        Img = SrcImg.copy()
        Canvas = SrcCanvas.copy()
        IPT.drawPoints(Img, np.array(LeftPoint).reshape(-1, 1), (0,0,255), radius=3)
        IPT.drawPoints(Canvas, np.array(RightPoint).reshape(-1, 1), (0,0,255), radius=3)
        for i in SrcPoints:
            IPT.drawPoints(Img, i.reshape(-1, 1), (0,255,0), radius=7)
        for i in CanvasPoints:
            IPT.drawPoints(Canvas, i.reshape(-1, 1), (0,255,0), radius=7)

        cv.imshow('Src', Img)
        cv.imshow('Canvas', Canvas)
        Key = chr(cv.waitKey(30) & 255)
        if Key == 'l':
            if len(SrcPoints) < 3:
                SrcPoints.append(np.array(LeftPoint).reshape(-1))
        elif Key == 'r':
            if len(CanvasPoints) < 3:
                CanvasPoints.append(np.array(RightPoint).reshape(-1))
        elif Key == 'p' or First:
            First = False
            SrcPointsA = np.array(SrcPoints, dtype=np.float32)
            CanvasPointsA = np.array(CanvasPoints, dtype=np.float32)
            print('SrcPoints:', SrcPointsA)
            print('CanvasPoints:', CanvasPointsA)
            PerspectiveMatrix = cv.getPerspectiveTransform(np.array(SrcPointsA),
                                                            np.array(CanvasPointsA))
            print('PerspectiveMatrix:\n', PerspectiveMatrix)
            PerspectiveImg = cv.warpPerspective(Img, PerspectiveMatrix, (300, 300))
            cv.imshow('PerspectiveImg', PerspectiveImg)
            cv.imwrite('../../../Datas/Output/PerspectiveImg.png', PerspectiveImg)
            cv.imwrite('../../../Datas/Output/Paper.jpg', Img)
        elif Key == 'f':
            SrcPoints = []
            print('reset SrcPoints')
        elif Key == 'f':
            CanvasPoints = []
            print('reset CanvasPoints')
        elif Key == 'q':
            break
