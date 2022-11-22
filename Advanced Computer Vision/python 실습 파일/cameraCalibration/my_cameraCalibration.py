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

chessboardSize = (10, 7)    # [Q] Please discuss the meaning of the paramter    -> number of squares
## Sample 1
# frameSize = (2984, 3984)    # [Q] Please discuss the meaning of the paramter    -> frame size (width, height)
# frameSize = (1193, 1593)    # [Q] Please discuss the meaning of the paramter    -> frame size (width, height)
## Sample 2
frameSize = (1593, 1193)    # [Q] Please discuss the meaning of the paramter    -> frame size (width, height)



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)    # [Q] Please discuss why we need "criteria".
# 2 + 1, 30, 0.001 => iterations 30, epsillon(정확도) 0.001 


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) 

size_of_chessboard_squares_mm = 25 				# chessboard size (25mm*25mm)
objp = objp * size_of_chessboard_squares_mm     # [Q] Please discuss the meaning of the paramter. Specify the unit of it.
# (408, 3) 

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space  # [Q] How many points?
imgpoints = []  # 2d points in image plane.     # [Q] How many points?


# images = glob.glob('./samples/*.png')           # [Q] How many images? 21
# images = glob.glob('./my_samples_1_1/*.jpg')        # [Q] How many images? 10
images = glob.glob('./my_samples_2/*.jpg')        # [Q] How many images? 8
# pdb.set_trace()

for image in images:
 
    img = cv.imread(image)  					
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)	

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)     # [Q] Please discuss the role of "findChessboardCorners()". Specify the meaning of its outputs.
    # corners에 corner 저장 (408, 1, 2) => 408 corners / when find everything ret == True

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(1000) 		# 1000 millisecond (1second) wait
    # pdb.set_trace()

cv.destroyAllWindows()
# pdb.set_trace()



############################ CALIBRATION ############################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
pdb.set_trace()
'''
### Sample 1 (2984, 3984)
# ret : 1.014401427981003
# cameraMatrix : 
    array([[3.09303222e+03, 0.00000000e+00, 1.47544772e+03],
           [0.00000000e+00, 3.09381436e+03, 1.98090220e+03],
           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# dist (distortion 왜곡계수) :
    array([[ 2.33714756e-01, -3.52242605e+00,  1.96244763e-03,
            -1.37751222e-03,  1.66613592e+01]])
    == (k1, k2, p1, p2, k3)
# rvecs (회전 벡터) (array([[-0.31600054],[-0.08009399],[-0.64537179]]), ...) (len=10)
# tvecs (변환 벡터) (array([[-125.77377811],[ -11.35885282],[ 624.9931446 ]]), ...) (len=10)

### Sample 1-1 (1193, 1593) - 해상도 줄임 
# ret : 0.4076682030341997
# cameraMatrix : 
    array([[1.23610062e+03, 0.00000000e+00, 5.87637498e+02],
           [0.00000000e+00, 1.23646818e+03, 7.90863971e+02],
           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# dist (distortion 왜곡계수) :
    array([[ 2.40087576e-01, -3.50662810e+00,  1.69234354e-03,
            -2.05372274e-03,  1.54986095e+01]])
    == (k1, k2, p1, p2, k3)
# rvecs (회전 벡터) (array([[-0.31684132],[-0.07876741],[-0.64510916]]), ...) (len=10)
# tvecs (변환 벡터) (array([[-124.77551631],[ -10.90340074],[ 625.10047509]]), ...) (len=10)


### Sample 2 (1593, 1193)
# ret : 0.3226935639005626
# cameraMatrix : 
    array([[1.23969856e+03, 0.00000000e+00, 7.84591802e+02],
           [0.00000000e+00, 1.23622046e+03, 5.95478344e+02],
           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# dist (distortion 왜곡계수) :
    array([[ 1.26502597e-01, -1.12168966e+00,  5.49239276e-03,
            -1.69399049e-04,  2.48314927e+00]])
    == (k1, k2, p1, p2, k3)
# rvecs (회전 벡터) (array([[-0.52124029],[ 0.26359321],[ 0.72538584]]), ...) (len=8)
# tvecs (변환 벡터) (array([[  4.28745803],[-81.14224314],[701.3190992 ]]), ...) (len=8)

    *retval: Average RMS re-projection error. This should be as close to zero as possible. 0.1 ~ 1.0 pixels in a good calibration.

    [Q] Please specify the focal length (fx, fy) and the principal point (cx, cy). fx=1.2396 fy=1.2362 / cx=7.8459 cy=5.9547

    [Q] Please specify the radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients.

    *Tangential distortion occurs because the image-taking lens is not aligned perfectly parallel to the imaging plane.

    [Q] Please discuss the meaning of "rvecs" and "tvecs". rotation vector 3x1, Translation vector 3x1

'''


############################ UNDISTORTION ############################
# img = cv.imread('./my_samples_1_1/LRM_20221004_143416.jpg')
img = cv.imread('./my_samples_2/LRM_20221004_135012.jpg')
h, w = img.shape[:2] 	
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
'''
    [Q] Please discuss the role of "getOptimalNewCameraMatrix". Why do we need this? Output new camera intrinsic matrix.
		프리 스케일링 매개변수를 기반으로 카메라 매트릭스를 개선. 결과를 자르는 데 사용할 수 있는 이미지 ROI를 반환

    [Q] What is roi"? Optional output rectangle that outlines all-good-pixels region in the undistorted image. 

'''


# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)


# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibResult_1_1(1).png', dst)


# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5) 	# Computes the undistortion and rectification transformation map. 
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)    # [Q] Please discuss the role of "remap()"  -> Applies a generic geometrical transformation to an image
# 왜곡된 이미지에서 왜곡되지 않은 이미지로의 매핑 함수를 찾고 remap 함수 사용 

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibResult_1_1(2).png', dst)

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

''' total error

## Sample 1_1
total error: 0.04745484988031208

## Sample 2
total error: 0.03766054588135971


'''