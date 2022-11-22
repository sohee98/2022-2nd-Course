'''
    Seokju Lee @ EE7107
    2022.09.16.

    Original code from 
    https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
    and
    https://github.com/niconielsen32/ComputerVision

    Korean reference
    https://leechamin.tistory.com/345
    
    Prerequisite:
    which pip
    pip install opencv-contrib-python

    If you want to experience VirtualCam, please follow the below instruction.
    https://github.com/kaustubh-sadekar/VirtualCam
'''

import numpy as np
import cv2 as cv
import glob
import pdb



################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS ################

chessboardSize = (24, 17)   # [Q] Please discuss the meaning of the paramter    -> number of squares
frameSize = (1440, 1080)    # [Q] Please discuss the meaning of the paramter    -> frame size



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)    # [Q] Please discuss why we need "criteria".
# 2 + 1, 30, 0.001 => iterations 30, epsillon(정확도) 0.001 


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) 

size_of_chessboard_squares_mm = 20 				# chessboard size (20mm*20mm)
objp = objp * size_of_chessboard_squares_mm     # [Q] Please discuss the meaning of the paramter. Specify the unit of it.
# (408, 3) 

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space  # [Q] How many points?
imgpoints = []  # 2d points in image plane.     # [Q] How many points?


images = glob.glob('./samples/*.png')           # [Q] How many images? 21
# pdb.set_trace()

for image in images:
 
    img = cv.imread(image)  					# (1080, 1440, 3)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)	# (1080, 1440) 흑백으로 변환 

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)     # [Q] Please discuss the role of "findChessboardCorners()". Specify the meaning of its outputs.
    # corners에 corner 저장 (408, 1, 2) => 408 corners / when find everything ret == True

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000) 		# 1000 millisecond (1second) wait
    # pdb.set_trace()

cv.destroyAllWindows()
# pdb.set_trace()



############################ CALIBRATION ############################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
# pdb.set_trace()
'''
# ret : 1.8562855738028046
# cameraMatrix : 
	array([[1.17232635e+03, 0.00000000e+00, 7.42502162e+02],
    	   [0.00000000e+00, 1.17209523e+03, 5.90997158e+02],
       	   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# dist (distortion 왜곡계수) array([[-0.24740003,  0.14635807,  0.00079243, -0.00039257, -0.081573  ]]) == (k1, k2, p1, p2, k3)
# rvecs (회전 벡터) (array([[-0.01642824],[-0.00286532],[-1.59347947]]), ...) (len=21)
# tvecs (변환 벡터) (array([[-101.46773608],[ 526.79606106],[1402.2331006 ]]), ...) (len=21)

    *retval: Average RMS re-projection error. This should be as close to zero as possible. 0.1 ~ 1.0 pixels in a good calibration.

    [Q] Please specify the focal length (fx, fy) and the principal point (cx, cy). fx=1.1723 fy=1.1720 / cx=7.4250 cy=5.9099

    [Q] Please specify the radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients. k1=-0.2474 k2=0.1463 k3=-0.0515 / p1=0.0007 p2=-0.0003

    *Tangential distortion occurs because the image-taking lens is not aligned perfectly parallel to the imaging plane.

    [Q] Please discuss the meaning of "rvecs" and "tvecs". rotation vector 3x1, Translation vector 3x1

'''


############################ UNDISTORTION ############################

img = cv.imread('./samples/Image__2018-10-05__10-29-04.png')
h, w = img.shape[:2] 	# (1080, 1440)
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
'''
(Pdb) roi
(65, 73, 1320, 957)
(Pdb) newCameraMatrix
array([[976.81481934,   0.        , 749.14618672],
       [  0.        , 983.87805176, 601.97666045],
       [  0.        ,   0.        ,   1.        ]])

    [Q] Please discuss the role of "getOptimalNewCameraMatrix". Why do we need this? Output new camera intrinsic matrix.
		프리 스케일링 매개변수를 기반으로 카메라 매트릭스를 개선. 결과를 자르는 데 사용할 수 있는 이미지 ROI를 반환

    [Q] What is roi"? Optional output rectangle that outlines all-good-pixels region in the undistorted image. 

'''


# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)


# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibResult1.png', dst)


# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5) 	# Computes the undistortion and rectification transformation map. 
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)    # [Q] Please discuss the role of "remap()"  -> Applies a generic geometrical transformation to an image
# 왜곡된 이미지에서 왜곡되지 않은 이미지로의 매핑 함수를 찾고 remap 함수 사용 

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibResult2.png', dst)

'''
    [Q] Please discuss the difference between 'calibResult1.png' and 'calibResult2.png'.
    	'calibResult2.png' size is smaller. more crop (1255*884) < (1320*957)
'''


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)): 
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist) 	#Projects 3D points to an image plane. (408, 1, 2)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

'''
    [Q] What is the meaning of "Reprojection Error"? Please specify the unit of it.
    	-> estimate if the parameters are correct. closer to zero, more accurate.  
'''

pdb.set_trace()

'''
(Pdb) imgpoints[20][0]
array([[254.72742, 246.56549]], dtype=float32)
(Pdb) imgpoints2[0]
array([[254.76506, 246.76791]], dtype=float32)
'''