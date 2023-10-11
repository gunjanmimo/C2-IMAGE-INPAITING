import cv2
import numpy as np
import sol_Laplace_Equation_Axb

from dataclasses import dataclass

@dataclass
class Parameters:
    hi: float
    hj: float
    dt: float
    iterMax: float
    tol: float


#Example script: You should replace the beginning of each function ('sol')
#with the name of your group. i.e. if your gropu name is 'G8' you should
#call :
#G8_DualTV_Inpainting_GD(I, mask, paramInp, paramROF)


# ======>>>>  input data  <<<<=======
# folder with the images
folderInput='/Users/angelicaatehortua/Documents/posdoctorado/sueltos/solution/'


#There are several black and white images to test:
#  image1_toRestore.jpg
#  image2_toRestore.jpg
#  image3_toRestore.jpg
#  image4_toRestore.jpg
#  image5_toRestore.jpg

# figure name to process. Options: image1 to image5
figure_name = 'image1'

# ======>>>>  process  <<<<=======

# read an image
figure_name_final=folderInput+figure_name+'_toRestore.jpg'
I = cv2.imread(figure_name_final,cv2.IMREAD_UNCHANGED)

#Number of pixels for each dimension, and number of channles
# get dimensions of image
dimensions = I.shape

# height, width, number of channels in image
height = I.shape[0]
width = I.shape[1]

print('Image Dimension    : ', dimensions)
print('Image Height       : ', height)
print('Image Width        : ', width)

# show image
cv2.imshow('image',I)
cv2.waitKey(0)

# Normalize values into [0,1]

min_val = np.min(I.ravel())
max_val = np.max(I.ravel())
I = (I.astype('float') - min_val)
I = I/max_val

# show normalized image
cv2.imshow('Normalized image',I)
cv2.waitKey(0)

# visualize the normalized image
cv2.imshow('Normalized Image', I)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Image data after Normalize:\n", I)

#Load the mask
mask_img_name=folderInput+figure_name+'_mask.jpg'
mask_img = cv2.imread(mask_img_name,cv2.IMREAD_UNCHANGED)

cv2.imshow('mask_img', mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# height, width, number of channels in image
height_mask = mask_img.shape[0]
width_mask = mask_img.shape[1]
dimensions_mask = mask_img.shape

ni=height_mask
nj=width_mask

print('Mask Dimension    : ', dimensions_mask)
print('Mask Height       : ', height_mask)
print('Mask Width        : ', width_mask)

#We want to inpaint those areas in which mask == 1
mask1 = mask_img >128
mask=mask1.astype('float')

#mask(i,j) == 1 means we have lost information in that pixel
#mask(i,j) == 0 means we have information in that pixel

# visualize the mask
cv2.imshow('mask>128', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  Parameters

param = Parameters(0,0,0,0,0)
param.hi = 1 / (ni-1)
param.hj = 1 / (nj-1)

#  Parameters for gradient descent (you do not need for week1)
param.dt = 5*10^-7
param.iterMax = 10^4
param.tol = 10^-5

# visualize the mask
cv2.imshow('Before', I)
cv2.waitKey(0)
cv2.destroyAllWindows()

u = sol_Laplace_Equation_Axb.sol_Laplace_Equation_Axb(I, mask, param)

# visualize the final image
cv2.imshow('After', u)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  Challenge image. (We have lost 99% of information)
del I, u, mask_img
figure_name='image6'
figure_name_final=folderInput+figure_name+'_toRestore.tif'
I = cv2.imread(figure_name_final,cv2.IMREAD_UNCHANGED)

#Normalize values into [0,1]

min_val = np.min(I.ravel())
max_val = np.max(I.ravel())
I = (I.astype('float') - min_val)
I = I/max_val

cv2.imshow('image6', I)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Number of pixels for each dimension, and number of channels

# height, width, number of channels in image
ni = I.shape[0]
nj = I.shape[1]
nC = I.shape[2]


figure_name='image6'
figure_name_final=folderInput+figure_name+'_mask.tif'
mask_img = cv2.imread(figure_name_final,cv2.IMREAD_UNCHANGED)

mask1 = mask_img >128
mask=mask1.astype('float')


#mask(i,j) == 1 means we have lost information in that pixel
#mask(i,j) == 0 means we have information in that pixel

# visualize the mask
cv2.imshow('mask>128', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

param = Parameters(0,0,0,0,0)
param.hi = 1 / (ni-1)
param.hj = 1 / (nj-1)


# visualize the image
cv2.imshow('before', I)
cv2.waitKey(0)
cv2.destroyAllWindows()

Iinp = np.zeros(I.shape, dtype=np.float)
Iinp[:,:,0]=sol_Laplace_Equation_Axb.sol_Laplace_Equation_Axb(I[:,:,0], mask[:,:,0], param)
Iinp[:,:,1]=sol_Laplace_Equation_Axb.sol_Laplace_Equation_Axb(I[:,:,1], mask[:,:,1], param)
Iinp[:,:,2]=sol_Laplace_Equation_Axb.sol_Laplace_Equation_Axb(I[:,:,2], mask[:,:,2], param)

# visualize the image
cv2.imshow('After', Iinp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Goal Image
del I, mask_img, Iinp

# Read the image
figure_name='image'
figure_name_final=folderInput+figure_name+'_to_Restore.png'
I = cv2.imread(figure_name_final,cv2.IMREAD_UNCHANGED)
Iinp = np.zeros(I.shape, dtype=np.float)

# height, width, number of channels in image
ni = I.shape[0]
nj = I.shape[1]
nC = I.shape[2]

#Normalize values into [0,1]

min_val = np.min(I.ravel())
max_val = np.max(I.ravel())
I = (I.astype('float') - min_val)
I = I/max_val

# We want to inpaint those areas in which mask == 1(red part of the image)

I_ch1 = I[:,:, 0]
I_ch2 = I[:,:, 1]
I_ch3 = I[:,:, 2]

# BGR in Python

#TO COMPLETE 1
mask = ????
#mask_img(i,j) == 1 means we have lost information in that pixel
#mask(i,j) == 0 means we have information in that pixel

# visualize the mask
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



#  Parameters for gradient descent(you do not need for week1)
# param.dt = 5 * 10 ^ -7;
# param.iterMax = 10 ^ 4;
# param.tol = 10 ^ -5;

# parameters
param.hi = 1 / (ni - 1)
param.hj = 1 / (nj - 1)

# for each channel

# visualize the image
cv2.imshow('Before', I)
cv2.waitKey(0)
cv2.destroyAllWindows()

Iinp[:,:, 0]=sol_Laplace_Equation_Axb.sol_Laplace_Equation_Axb(I_ch1, mask, param)
Iinp[:,:, 1]=sol_Laplace_Equation_Axb.sol_Laplace_Equation_Axb(I_ch2, mask, param)
Iinp[:,:, 2]=sol_Laplace_Equation_Axb.sol_Laplace_Equation_Axb(I_ch3, mask, param)

# visualize the image
cv2.imshow('After', Iinp)
cv2.waitKey(0)
cv2.destroyAllWindows()

