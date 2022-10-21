'''
    22.09.27. Seokju Lee
    Modified for python3
    Original code from https://github.com/Leo-LiHao/OpenCV-Python-Tutorials
'''


import sys
sys.path.append('../../../')

import cv2 as cv
import numpy as np
import pdb

from Src.ImageProcessing.Contours.ContourAnalyst import ContourAnalyst
import Src.ToolBox.ImageProcessTool as IPT


RED   = (0, 0, 255)
BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)


class GenChessboard(object):
    """
    **Function:**

        * :py:func:`drawChessboard`
        ---------------------------------------
        * :py:func:`generateCB`
        ---------------------------------------

    **Parameters:**
        * :py:attr:`BlockSize` type : :py:class:`int`, chessboard block size
        -----------------------------------------------------------
        * :py:attr:`BoardDim_ls` type : :py:class:`list`, chessboard dimensionality, the size is Nx2, the format is [[12, 13], [2, 3]]
        -----------------------------------------------------------
        * :py:attr:`OriginPos_ls` type : :py:class:`list`, chessboard origin position, the size is Nx2, the format is [[10,10], [100, 100]]
        -----------------------------------------------------------
        * :py:attr:`BackGroundValue` type : :py:class:`int`, the value for white background
        -----------------------------------------------------------
        * :py:attr:`ScreenPixel` type : :py:data:`tuple`, screen pixel
        -----------------------------------------------------------
    """

    def __init__(self,  BoardDim_ls, BlockSize, OriginPos_ls, BackGroundValue=150):
        self.BlockSize = BlockSize
        self.BoardDim_ls = BoardDim_ls
        self.OriginPos_ls = OriginPos_ls
        self.BackGroundValue = BackGroundValue
        self.ScreenPixel = (210, 210)

    def drawChessboard(self, BoardDim, OriginPos=[0,0], Img=None):
        """
        This function is for drawing a chessboard in the img.\n
        :param BoardDim: :py:class:`list`, chessboard dimension
        :param OriginPos: :py:class:`list`, origin position of the chessboard
        :param Img: :py:class:`ndarray`, the image usd to draw a chessboard, default is None
        :return: Img, :py:class:`numpy.ndarray`, the image have been draw a chessboard
        """

        ############ generate a new white image for drawing chessboard, and its size is the screen pixel
        if Img == None:
            Img = np.ones((self.ScreenPixel[1], self.ScreenPixel[0], 3), dtype=np.uint8)
            Img[:] = self.BackGroundValue
        ############# draw one chessboard
        if self.BlockSize != 0 and 0 not in BoardDim:
            for j in range(BoardDim[1]+1):
                for i in range(BoardDim[0]+1):
                    if (i + j) % 2 != 0:
                        x1 = OriginPos[0] + i * self.BlockSize
                        x2 = OriginPos[0] + (i + 1) * self.BlockSize
                        y1 = OriginPos[1] + j * self.BlockSize
                        y2 = OriginPos[1] + (j + 1) * self.BlockSize
                        Img[x1: x2, y1: y2, :] = 0
        return Img

    def generateCB(self):
        """
        This function is for generate a image painted with chessboards.\n
        :return: showImg, :py:class:`numpy.ndarray`, image with chessboards on it
        """
        assert len(self.OriginPos_ls) == len(self.BoardDim_ls)
        showImg = self.drawChessboard(self.BoardDim_ls[0],self.OriginPos_ls[0],  None)
        ######## draw N chessboard
        for i in range (1,len(self.BoardDim_ls)):
            showImg = self.drawChessboard(self.BoardDim_ls[i], self.OriginPos_ls[i], showImg)
        return showImg


if __name__ == '__main__':
    '''
        We have two image inputs: 1) Chssboard, 2) Lena.
    '''

    # FilePath = '../../../Datas/lena.png'
    FilePath = '../../../Datas/dog_1.png'
    CB = GenChessboard(BoardDim_ls=[[7, 7]], BlockSize=25, OriginPos_ls=[[5, 5]], BackGroundValue=255)
    ChessBoardImg = CB.generateCB()
    SrcImg = cv.imread(FilePath)
    # blur
    SrcImg = cv.blur(SrcImg, (3,3))
    # SrcImg = ChessBoardImg
    GrayImg = cv.cvtColor(src=SrcImg, code=cv.COLOR_BGR2GRAY)

    '''
        [Q] Please briefly discuss the meaning of the Formula below.
            lambda_1과 lambda_2가 둘다 작으면 flat region, 하나가 더 크면 edge region, 둘다 크면 corner region으로 판정한다.
            이를 수식으로 만들어서 R 값을 구하는 공식이다.
            R이 T보다 크면 corner로 판정하고, 작으면 edge로 판정할 수 있도록 한다.
    '''

    # ----------------------- Formula ----------------------- #
    # R = det(M) - k*(trace(M))**2
    # where det(M) = lambda_1 * lambda_2
    #       trace(M) = lambda_1 + lambda_2
    #       lambda_1 and lambda_2 are the eigen values of M
    # ----------------------- Formula ----------------------- #

    NeighbourSize = 2
    SobelKernelSize = 3
    HarrisResult = \
        cv.cornerHarris(src=GrayImg.astype(np.float32), blockSize=NeighbourSize, ksize=SobelKernelSize, k=0.04)
    ThreshPercent = 0.01
    HarrisResultLessZero = HarrisResult[HarrisResult<0]
    # EdgeThresh = (HarrisResult.min() * ThreshPercent)
    # EdgeThresh = HarrisResultLessZero.mean() - 3*HarrisResultLessZero.std()
    EdgeThresh = 0
    Edge = HarrisResult < (EdgeThresh)
    Corner = HarrisResult > (HarrisResult.max() * ThreshPercent)
    Flat = ~(Corner | Edge)

    '''
        [Q] Please discuss how Edge, Corner, and Flat are defined.
            HarrisResult가 EdgeThresh(0)보다 작으면 edge, 
            최대값*0.01보다 크면 corner,
            나머지는 Flat으로 define
    '''

    Canvas = np.zeros(SrcImg.shape, dtype=np.uint8)
    Canvas[Corner] = RED
    Canvas[Edge] = GREEN
    Canvas[Flat] = BLUE

    '''
        [Q] Please discuss why we need Canvas.
            To see edge and corner in one image. Corner = red, Edge = green, Flat = blue

    '''

    # get corner points
    CornerImg = np.zeros(GrayImg.shape, dtype=np.uint8)
    CornerImg[Corner] = 255
    cv.imshow('CornerImg', CornerImg)
    CornerImg = cv.morphologyEx(src=CornerImg, op=cv.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.uint8))   # Dilation followed by Erosion. remove noise (배경에 흰색 점들)
    Contours, _ = cv.findContours(image=CornerImg, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    Corners_2xn = np.array([0, 0]).reshape(2, 1)
    for contour in Contours:
        CornerPt_2x1 = ContourAnalyst.getCentroid(contour=contour)
        Corners_2xn = np.hstack((Corners_2xn, CornerPt_2x1))
    Corners_2xn = Corners_2xn[:, 1:]
    ShowImg = SrcImg.copy()
    IPT.drawPoints(img=ShowImg, pts_2xn=Corners_2xn, color=RED)

    '''
        [Q] Please discuss the process of finding corner points.
            CornerImg에 corner를 흰색으로 찍은 후 (=255), cv.morphologyEx 로 remove noise (배경에 흰색 점들 제거)
            find contours : 검정 배경에서 흰색 점들 찾기

        [Q] Why do we need sub pixel?
            cv.cornerSubPix : Refines the corner locations

    '''

    # sub pixel
    IterationCounts = 100
    EpsilonValue = 0.001
    Criteria = (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, IterationCounts, EpsilonValue)
    CornersPts_nx1x2 = Corners_2xn.T.reshape(-1, 1, 2).astype(np.float32).copy()
    cv.cornerSubPix(image=GrayImg, corners=CornersPts_nx1x2, winSize=(5, 5), zeroZone=(-1, -1), criteria=Criteria)
    print('CornerNumber: ', CornersPts_nx1x2.shape[0])
    IPT.drawPoints(img=ShowImg, pts_2xn=CornersPts_nx1x2.T.reshape(2, -1), color=GREEN)
    cv.namedWindow('SrcImg', cv.WINDOW_NORMAL)
    cv.imshow('Canvas', Canvas)
    cv.imshow('SrcImg', ShowImg)
    cv.waitKey()

    '''
        [Q] Please compare the results of 1) Chessboard and 2) Lena.
            chessboard : corner가 규칙적이고, cv.morphologyEx를 해도 지울게 없었음. 
            Lena는 corner가 매우 많고 뷸규칙적이다.

        [Q] Please try other internet images with "cv.blur()"" function (comparing images with different blur kernels).
            # blur
            SrcImg = cv.blur(SrcImg, (3,3))

    '''    

    pdb.set_trace()