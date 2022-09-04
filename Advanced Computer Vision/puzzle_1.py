'''
  Seokju Lee
  2022. 04. 05. Image Processing Puzzle 1
'''

import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from PIL import Image
import pdb
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None' 




def show_image(i, img):
  if len(img.shape) == 2:
    plt.figure(i); plt.imshow(img, cmap='gray'); plt.xticks([]); plt.yticks([]); plt.ion(); plt.show()
  else:
    plt.figure(i); plt.imshow(img); plt.xticks([]); plt.yticks([]); plt.ion(); plt.show()


def close_image(i):
  if i == 0:
    plt.close('all')
  else:
    plt.close(i)


def check_same(a, b):
  if np.abs(a-b).mean() == 0:
    print('Mission complete!')
  else:
    print('Inputs are different. Try again!')





###################################################################################################
'''
  ### Discussion 1 ###
  1.1. Please check the shape of 'img1'.
  1.2. Please visualize an image.
  1.3. Please discuss the roles of three predefined functions.
  1.4. Use comment lines as needed.
'''

### Load image ###
im = data.coffee() / 255.    # Why do we divide image by '255.'?

### Check the shape of 'im' ###
# print("Shape of 'im':", im.shape)  # What is the shape of 'im'? Why there are three channels?

### Visualize 'im' ###
# show_image(1, im)  # Why do we input '1'?

###################################################################################################



###################################################################################################
'''
  ### Discussion 2: shifting image ###
  2.1. Please load a ground truth (GT) array from 'samples/gt1.npy'
  2.2. Please convert 'im' to look like 'gt1' by using array shifting.
  2.3. Please load a ground truth (GT) array from 'samples/gt2.npy'
  2.4. Please convert 'im' to look like 'gt2' by using array shifting.
'''

### Close all images ###
# close_image(0)

### Load GT ###
gt1 = np.load('samples/gt1.npy')

### Visualize 'gt1' and 'im' ###
# show_image(1, gt1)    # Please check the values of black pixels
# show_image(2, im)

### Please convert 'im' to look like 'gt1' ###
im1 = np.zeros(im.shape)  # What is the np.zeros function? What is the shape of 'im1'?

### Your implementation goes from here... ###

### Did you make it by trial-and-error? Please try the LOOP! ###

### Check whether they are the same ###
check_same(gt1, im1)


### Try 'samples/gt2.npy' ###
### Load GT ###
gt2 = np.load('samples/gt2.npy')
im2 = np.zeros(im.shape)

### Your implementation goes from here... ###

### Check whether they are the same ###
check_same(gt2, im2)

###################################################################################################



###################################################################################################
'''
  ### Discussion 3: grayscale image ###
  3.1. Please load a ground truth (GT) array from 'samples/gt3.npy'
  3.2. Please convert 'im' to look like 'gt3' by using np.mean function.
'''
### Close all images ###
# close_image(0)

### Load GT ###
gt3 = np.load('samples/gt3.npy')

### Check the shape of gt3 ###
# print("Shape of 'gt3':", gt3.shape)  # What is the shape of 'gt3'?

### Visualize gt3 ###
# show_image(1, gt3)  # What is the definition of grayscale image?

### Please convert 'im' to look like 'gt3' ###
im3 = np.zeros(im[:,:,0].shape)

### Check whether they are the same ###
check_same(gt3, im3)

###################################################################################################







pdb.set_trace()
