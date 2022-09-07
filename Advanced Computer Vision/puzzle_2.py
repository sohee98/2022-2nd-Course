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
  # plt.colorbar()
  # plt.grid()
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
  4.3. What is broadcasting?
  4.4. What are the rules for allowing broadcasting?
'''
### Close all images ###
close_image(0)

### Load image ###
im = data.cat() / 255.  # Why do we divide image by '255.'?
gt1 = np.load('samples/puzzle2/gt1.npy')
# pdb.set_trace()

### Check the shape of 'im' ###
# print("Shape of 'im':", im.shape)  # What is the shape of 'im'? Why are there three channels?

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

### Multiply the binary mask with im ###
### 'im' and 'mask' have different shape. How is this possible? Please check BROADCASTING! ###
# im1 = im * mask

# check_same(gt1, im1)
###################################################################################################


###################################################################################################
'''
  ### Discussion 5: Image cropping and resizing ###
  5.1. Crop the image (cat's eye) by array slicing.
  5.2. Increase the size of the "cat's eye" image to 400 x 400. Please discuss the difference between "rescale" and "resize".
'''
### Close all images ###
# close_image(0)

# im2 = im[y1:y2, x1:x2]
# print("Shape of 'im2':", im2.shape)

### Two different resizing function from scikit-image library ###
### Please discuss the meaning of input tuples ##
# im2_1 = rescale(im2, (4,4,1))
# im2_2 = resize(im2, (400,400))

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

# im = imageio.imread('samples/puzzle2/random_face.png') / 255.  # Download your own image
# im = resize(im, (300,300))
# show_image(1, im)

### Apply bilateral filter ###
# im_b = restoration.denoise_bilateral(im, sigma_color=0.03, sigma_spatial=2, channel_axis=-1)
# show_image(2, im_b)

# pdb.set_trace()
###################################################################################################





