'''
  Seokju Lee
  2022. 04. 12. Image Processing Puzzle 2
'''

import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage.transform import rescale, resize
from skimage import restoration
import imageio
import time
import pdb

def show_image(i, img):
  if len(img.shape) == 2:
    plt.figure(i); plt.imshow(img, cmap='gray');
  else:
    plt.figure(i); plt.imshow(img); 
  # plt.xticks([]); plt.yticks([]);
  plt.colorbar()
  plt.grid()
  plt.tight_layout()
  plt.ion()
  plt.show()


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


###################################################################################################
'''
  ### Discussion 4 - Image multiplication ###
  4.1. Please load a ground truth (GT) array from 'samples/puzzle2/gt1.npy'
  4.2. Please convert 'im' to look like 'gt1' by multiplying the binary mask.
  4.3. What is broadcasting? -> broadcasting = how NumPy treats arrays with different shapes during arithmetic operations
  4.4. What are the rules for allowing broadcasting? 
	1. If the arrays don’t have the same rank then prepend the shape of the lower rank array with 1s until both shapes have the same length.
	2. The two arrays are compatible in a dimension if they have the same size in the dimension or if one of the arrays has size 1 in that dimension.
	3. The arrays can be broadcast together iff they are compatible with all dimensions.
	4. After broadcasting, each array behaves as if it had shape equal to the element-wise maximum of shapes of the two input arrays.
	5. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension.
'''
### Close all images ###
close_image(0)

### Load image ###
im = data.cat() / 255.  # Why do we divide image by '255.'? normalization -> make values 0~1
gt1 = np.load('samples/puzzle2/gt1.npy')
pdb.set_trace()

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

# pdb.set_trace()

### Multiply the binary mask with im ###
### 'im' and 'mask' have different shape. How is this possible? Please check BROADCASTING! ### 
# Broadcasting = how NumPy treats arrays with different shapes during arithmetic operations
im1 = im * mask

check_same(gt1, im1)
###################################################################################################


###################################################################################################
'''
  ### Discussion 5: Image cropping and resizing ###
  5.1. Crop the image (cat's eye) by array slicing.
  5.2. Increase the size of the "cat's eye" image to 400 x 400. Please discuss the difference between "rescale" and "resize".
		rescale : change the size of whole image by resampling it. specify scaling factor 
		resize : change the size, dimensions(width, height) = extend the image. specify output image shape

'''
### Close all images ###
# close_image(0)

# pdb.set_trace()

im2 = im[y1:y2, x1:x2] 
# print("Shape of 'im2':", im2.shape)  	# (100, 100, 3)

### Two different resizing function from scikit-image library ###
### Please discuss the meaning of input tuples ##
im2_1 = rescale(im2, (4,4,1)) 		# (400, 400, 3) specify scaling factor 
im2_2 = resize(im2, (400,400)) 		# (400, 400, 3) specify output image shape

### Visualize the resized images ###
# show_image(1, im2)
# show_image(2, im2_1)
# show_image(3, im2_2)
###################################################################################################


###################################################################################################
'''
  ### Discussion 6: Photoshop - bilateral filter ###
  6.1. Please capture random faces from "https://generated.photos/face-generator/new"
  6.2. Please apply a "bilateral filter" on the image.
  6.3. Please analyze the effect of the bilateral filter by changing the input parameters.
  6.4. What are the pros and cons of filtering?

  * Bilateral filter: An edge-preserving and noise reducing filter by averaging pixels based on their spatial closeness and radiometric similarity.
'''
### Close all images ###
# close_image(0)
# pdb.set_trace()

im = imageio.imread('samples/puzzle2/random_face.png') / 255.  # Download your own image
im = resize(im, (300,300))
show_image(1, im)

### Apply bilateral filter ###
im_b = restoration.denoise_bilateral(im, sigma_color=0.03, sigma_spatial=2, channel_axis=-1)
show_image(2, im_b)

# sigma_color:blur effect, larger ->  larger radiometric diffferences
# sigma_spatial:larger -> an average of pixels with large spatial differences


pdb.set_trace()
###################################################################################################





