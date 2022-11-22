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

# chessboardSize = (10, 7)    # [Q] Please discuss the meaning of the paramter    -> number of squares
# chessboardSize = (7, 6)    
chessboardSize = (9, 6)    

# frameSize = (796, 597)    
frameSize = (960, 720)    



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)    # [Q] Please discuss why we need "criteria".
# 2 + 1, 30, 0.001 => iterations 30, epsillon(정확도) 0.001 


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) 

# size_of_chessboard_squares_mm = 25              # chessboard size (25mm*25mm)
size_of_chessboard_squares_mm = 7 				# chessboard size (25mm*25mm)
objp = objp * size_of_chessboard_squares_mm     # [Q] Please discuss the meaning of the paramter. Specify the unit of it.
# (408, 3) 

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space  # [Q] How many points?
imgpoints = []  # 2d points in image plane.     # [Q] How many points?


# images = glob.glob('./samples/*.png')           # [Q] How many images? 21
# images = glob.glob('./sfm_calib_images5/*.jpg')    
images = glob.glob('./calib_prof/*.jpg')    
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
### Images 2 (796, 596)    
# ret : 1.769461119729166
# cameraMatrix : 
    array([[1.61706747e+03, 0.00000000e+00, 4.41386469e+02],
           [0.00000000e+00, 2.86819874e+03, 4.19266253e+02],
           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# dist (distortion 왜곡계수) :
    array([[-2.38736698e+00,  4.32770085e+01, -8.18229511e-03,
            -5.74574940e-02, -3.91208336e+02]])
    == (k1, k2, p1, p2, k3)

### Images 3
# cameraMatrix :
    array([[670.76093669,   0.        , 296.53161507],
           [  0.        , 667.18551759, 401.80752139],
           [  0.        ,   0.        ,   1.        ]])
# dist
    array([[ 1.03833525e-01, -1.93640762e+00,  7.75480314e-03,
            -3.80608835e-04,  5.08940106e+00]])

### Images 4
# cameraMatrix :
    array([[658.77958917,   0.        , 299.36742118],
           [  0.        , 656.77876625, 394.23719598],
           [  0.        ,   0.        ,   1.        ]])
# dist
    array([[ 1.44126438e-01, -2.03790206e+00,  5.51810641e-03,
            -1.02301327e-03,  5.10080643e+00]])

### Images 5
# cameraMatrix :
    array([[651.53868284,   0.        , 299.46735177],
           [  0.        , 647.42821239, 397.80440433],
           [  0.        ,   0.        ,   1.        ]])
   array([[653.93843191,   0.        , 298.89034343],
           [  0.        , 650.60680188, 395.17532636],
           [  0.        ,   0.        ,   1.        ]])
   array([[657.78913549,   0.        , 300.0778785 ],
           [  0.        , 653.34350191, 398.70946196],
           [  0.        ,   0.        ,   1.        ]])
# dist
    array([[ 1.73967808e-01, -3.13328792e+00,  9.14688401e-03,
            -8.10229797e-04,  1.05022313e+01]])

### prof
array([[1.49085909e+03, 0.00000000e+00, 4.74412166e+02],
       [0.00000000e+00, 1.48819603e+03, 3.47256136e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
total error: 0.010135302553137672




    *retval: Average RMS re-projection error. This should be as close to zero as possible. 0.1 ~ 1.0 pixels in a good calibration.

    [Q] Please specify the focal length (fx, fy) and the principal point (cx, cy). fx=1.2396 fy=1.2362 / cx=7.8459 cy=5.9547

    [Q] Please specify the radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients.

    *Tangential distortion occurs because the image-taking lens is not aligned perfectly parallel to the imaging plane.

    [Q] Please discuss the meaning of "rvecs" and "tvecs". rotation vector 3x1, Translation vector 3x1

'''


############################ UNDISTORTION ############################
# img = cv.imread('./my_samples_1_1/LRM_20221004_143416.jpg')
# img = cv.imread('./sfm_calib_images4/LRM_20221108_134132.jpg')
# img = cv.imread('./sfm_calib_images5/LRM_20221108_140056.jpg')
img = cv.imread('./calib_prof/1667885355234.jpg')

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
cv.imwrite('calibResult_2(1).png', dst)


# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5) 	# Computes the undistortion and rectification transformation map. 
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)    # [Q] Please discuss the role of "remap()"  -> Applies a generic geometrical transformation to an image
# 왜곡된 이미지에서 왜곡되지 않은 이미지로의 매핑 함수를 찾고 remap 함수 사용 

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibResult_2(2).png', dst)

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

## Image 4
total error: 0.04550468552874256



'''