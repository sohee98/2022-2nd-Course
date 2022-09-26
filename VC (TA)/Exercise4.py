import numpy as np

## 1. Show Image

from skimage import data

### Load image ###
im = data.coffee() / 255.  # Why do we divide image by '255.'?
print(im)
### Check the shape of 'im' ###
print("Shape of 'im':", im.shape)  # What is the shape of 'im'? Why are there three channels?

from matplotlib import pyplot as plt
plt.imshow(im)
plt.show() 

import PIL.Image
im = PIL.Image.open('path/to/your/image')
im = np.array(im)

np.save(file, arr, allow_pickle=True, fix_imports=True)


#-----------------------------
## 2. Shift Image

# import numpy as np
# from matplotlib import pyplot as plt
from skimage import data
from PIL import Image
import pdb
import matplotlib as mpl

mpl.rcParams['toolbar'] = 'None'


def show_image(i, img):
    if len(img.shape) == 2:
        plt.figure(i); plt.imshow(img, cmap='gray'); plt.xticks([]); plt.yticks([]); plt.ion(); plt.show();
    else:
        plt.figure(i); plt.imshow(img); plt.xticks([]); plt.yticks([]); plt.ion(); plt.show();

def close_image(i):
    if i == 0:
        plt.close('all')
    else:
        plt.close(i)

def check_same(a, b):
    if np.abs(a - b).mean() == 0:
        print('Mission complete!')
    else:
        print('Inputs are different. Try again!')

### Load image ###
im = data.coffee()

### Load GT ###
gt1 = np.load('gt1.npy')

### Visualize 'gt1' and 'im' ###
show_image(0, gt1)  # Please check the values of black pixels
#show_image(1, im)

### Please convert 'im' to look like 'gt1' ###
im1 = np.zeros(im.shape)  # What is the np.zeros function? What is the shape of 'im1'?

### Your implementation goes from here... ###
hh, ww, cc = im.shape  		# hh=400, ww=600, cc=3

for i in range(ww):
	# print('index:', i)
	im1 = np.zeros(im.shape)
	im1[:, i:, :] = im[:, :ww-i, :]
	if (gt1 == im1).all():
		break

### Check whether they are the same ###
check_same(gt1, im1)


### Try 'samples/gt2.npy' ###
### Load GT ###
gt2 = np.load('gt2.npy')
im2 = np.zeros(im.shape)

### Your implementation goes from here... ###
gt2_H = gt2.sum(axis=(0,2)) 	# (600, ) heigth, channel 따라 값 더함  
gt2_V = gt2.sum(axis=(1,2))		# (400, ) width, channel 따라 값 더함  

jj = (gt2_H == 0).sum() 		# 100 (0이 아닌 index 찾기)
ii = (gt2_V == 0).sum() 		# 100

im2[ii:, jj:, :] = im[:hh-ii, :ww-jj, :]

### Check whether they are the same ###
check_same(gt2, im2)


#-----------------------------
## 3. Grayscale Image

### Load GT ###
gt3 = np.load('gt3.npy')

### Check the shape of gt3 ###
print("Shape of 'gt3':", gt3.shape)  # What is the shape of 'gt3'?

### Visualize gt3 ###
show_image(1, gt3)  # What is the definition of grayscale image?

### Please convert 'im' to look like 'gt3' ###
im3 = np.zeros(im[:, :, 0].shape)

## Put your code here
ww, hh, cc = im.shape
im3 = im.mean(axis=2)

### Check whether they are the same ###
check_same(gt3, im3)


#-----------------------------
## 4. Image Multiplication 

### Load image ###
im = data.cat() / 255.  # Why do we divide image by '255.'? normalization -> make values 0~1
gt1 = np.load('gt4.npy')

### Check the shape of 'im' ###
# print("Shape of 'im':", im.shape)  # What is the shape of 'im'? Why are there three channels? -> (300, 451, 3) RGB channels=3

### Visualize 'gt1' and 'im' ###
# show_image(1, im)
# show_image(2, gt1)

### Please convert 'im' to look like 'gt1' by multiplying binary mask. ###
### Hint: "The region of cat's eye has the size of 100 x 100." ###

### Make a binary mask first ###
# mask = np.ones(im.shape).mean(axis=2, keepdims=True)
# print("Shape of 'mask':", mask.shape) 
'''
  im1 : 
  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0
  0  0  P  1  1  1  0  0  0  0  0
  0  0  1  1  1  1  0  0  0  0  0
  0  0  1  1  1  1  0  0  0  0  0    -> Assign ones only in a specific area (cat's eye).
  0  0  1  1  1  Q  0  0  0  0  0    -> Find the coordinate value of P(x1,y1) and Q(x2,y2)
  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0

  Please make a cat's eye mask!
'''
mask = np.zeros(im.shape).mean(axis=2, keepdims=True) 	# (300, 451, 1)

hh, ww, cc = im.shape
for i in range(hh):
	for j in range(ww):
		if gt1[i, j, 0]:
			mask[i, j, 0] = 1
			y2, x2 = i, j 		# x2, y2 = 219, 164
y1, x1 = y2-100, x2-100  		# x1, y1 = 119, 64 	(120, 65)

### Multiply the binary mask with im ###
### 'im' and 'mask' have different shape. How is this possible? Please check BROADCASTING! ### 
# Broadcasting = how NumPy treats arrays with different shapes during arithmetic operations
im1 = im * mask

check_same(gt1, im1)

#-----------------------------
## 5. Image cropping and resizing

from skimage.transform import rescale, resize

im2 = im[y1:y2, x1:x2]
# print("Shape of 'im2':", im2.shape)

### Two different resizing function from scikit-image library ###
### Please discuss the meaning of input tuples ##
im2_1 = rescale(im2, (4,4,1))
im2_2 = resize(im2, (400,400))
'''	rescale : change the size of whole image by resampling it. specify scaling factor 
    resize : change the size, dimensions(width, height) = extend the image. specify output image shape'''

### Visualize the resized images ###
# show_image(1, im2)
# show_image(2, im2_1)
# show_image(3, im2_2)


#-----------------------------
## 6. Photoshop - bilateral filter
im = imageio.imread('random_face.png') / 255.  # Download your own image
im = resize(im, (300,300))
show_image(1, im)

### Apply bilateral filter ###
im_b = restoration.denoise_bilateral(im, sigma_color=0.03, sigma_spatial=2, channel_axis=-1)
show_image(2, im_b)

# sigma_color:blur effect, larger ->  larger radiometric diffferences
# sigma_spatial:larger -> an average of pixels with large spatial differences

