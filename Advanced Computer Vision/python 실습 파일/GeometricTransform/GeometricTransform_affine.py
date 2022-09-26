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
    SrcImg = cv.imread('../../../Datas/AffineLena.png')
    SrcCanvas = np.zeros(SrcImg.shape, dtype=np.uint8)
    # cv.namedWindow("Src", cv.WINDOW_NORMAL)
    cv.namedWindow("Src")
    cv.setMouseCallback("Src", OnMouse1)
    cv.namedWindow("Canvas")
    # cv.namedWindow("Canvas", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Canvas", OnMouse2)

    SrcPoints = np.float32([[150,123],[439,174],[148,380]])
    CanvasPoints = np.float32([[0,0],[512,0],[0,512]])
    while True:
        Img = SrcImg.copy()
        Canvas = SrcCanvas.copy()
        IPT.drawPoints(Img, np.array(LeftPoint).reshape(-1, 1), (0,0,255), radius=3)
        IPT.drawPoints(Canvas, np.array(RightPoint).reshape(-1, 1), (0,0,255), radius=3)
        for i in SrcPoints:
            IPT.drawPoints(Img, i.reshape(-1, 1), (0,255,0), radius=3)
        for i in CanvasPoints:
            IPT.drawPoints(Canvas, i.reshape(-1, 1), (0,255,0), radius=3)

        cv.imshow('Src', Img)
        cv.imshow('Canvas', Canvas)
        Key = chr(cv.waitKey(30) & 255)
        if Key == 'l':
            if len(SrcPoints) < 3:
                SrcPoints.append(np.array(LeftPoint).reshape(-1))
        elif Key == 'r':
            if len(CanvasPoints) < 3:
                CanvasPoints.append(np.array(RightPoint).reshape(-1))
        elif Key == 'a' or First:
            First = False
            SrcPointsA = np.array(SrcPoints, dtype=np.float32)
            CanvasPointsA = np.array(CanvasPoints, dtype=np.float32)
            print('SrcPoints:', SrcPointsA)
            print('CanvasPoints:', CanvasPointsA)
            AffineMatrix = cv.getAffineTransform(np.array(SrcPointsA), np.array(CanvasPointsA))
            print('AffineMatrix:\n', AffineMatrix)
            AffineImg = cv.warpAffine(Img, AffineMatrix, (Img.shape[1], Img.shape[0]))
            cv.imshow('AffineImg', AffineImg)
            cv.imwrite('../../../Datas/Output/InvAffineLena.png', AffineImg)
        elif Key == 'f':
            SrcPoints = []
            print('reset SrcPoints')
        elif Key == 'r':
            CanvasPoints =[]
            print('reset CanvasPoints')
        elif Key == 'q':
            break
