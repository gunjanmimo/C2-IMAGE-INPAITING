# TEAM 4 WEEK 1

import cv2
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sol_Laplace_Equation_Axb import sol_Laplace_Equation_Axb
import os

# Parameter class
@dataclass
class Parameters:
    hi: float
    hj: float
    dt: float
    iterMax: float
    tol: float

# Function to perform inpainting
def image_inpainting(I, mask_image, result_folder, image_name):
    if mask_image is None:
        print(f"Error: Mask image for {image_name} not found or couldn't be read.")
        return I

    min_val = np.min(I.ravel())
    max_val = np.max(I.ravel())
    I = (I.astype("float") - min_val) / max_val

    ni, nj = I.shape[0], I.shape[1]

    mask1 = mask_image > 128
    mask = mask1.astype("float")

    param = Parameters(0, 0, 0, 0, 0)
    param.hi = 1 / (ni - 1)
    param.hj = 1 / (nj - 1)

    if len(I.shape) == 3:
        Iinp = np.zeros(I.shape, dtype=np.float32)
        for channel in range(I.shape[2]):
            Iinp[:, :, channel] = sol_Laplace_Equation_Axb(
                I[:, :, channel], mask[:, :, channel], param
            )
    else:
        Iinp = sol_Laplace_Equation_Axb(I, mask, param)

    # Save the inpainted image
    result_path = os.path.join(result_folder, f"{image_name}_inpaint.jpg")
    I_save = cv2.cvtColor(
        (Iinp * max_val + min_val).astype(np.uint8), cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(result_path, I_save)
    # cv2.imwrite(result_path, Iinp)

    # Display the original and inpainted images
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
    axarr[0].imshow(I)
    axarr[0].axis("off")
    axarr[0].set_title("Before inpainting")

    axarr[1].imshow(Iinp)
    axarr[1].axis("off")
    axarr[1].set_title("After inpainting")

    plt.tight_layout()
    plt.show()

# Paths and filenames
folderInput = '/Users/dianatat/Documents/Master/C2 Optimisation techniques for CV/Project/week1/code/images'
result_folder = '/Users/dianatat/Documents/Master/C2 Optimisation techniques for CV/Project/week1/code/images/results' 

image_names = [
    'image1',
    'image2',
    'image3',
    'image4',
    'image5',
]

for image_name in image_names:
    # Read the image to be restored
    image_path = f"{folderInput}/{image_name}_toRestore.jpg"
    I = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Read the corresponding mask
    mask_path = f"{folderInput}/{image_name}_mask.jpg"
    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Perform inpainting and save the result
    image_inpainting(I, mask_img, result_folder, image_name)


# image hola carola
figure_name_final = os.path.join(folderInput, "Image_to_Restore.png")
src_image = cv2.imread(figure_name_final)
src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

# Create mask
lower_red = np.array([0, 0, 0])
upper_red = np.array([255, 0, 0])

# Create the mask
mask = cv2.inRange(src_image, lower_red, upper_red)

# Convert the single channel mask to a 3 channel mask
mask_3_channel = cv2.merge([mask, mask, mask])

# Example image 1
figure_name_ex1 = os.path.join(folderInput, "example 1.jpg")
I_ex1 = cv2.imread(figure_name_ex1, cv2.IMREAD_UNCHANGED)
I_ex1 = cv2.cvtColor(I_ex1, cv2.COLOR_BGR2RGB)
I_ex1 = cv2.resize(I_ex1, (400, 400))

# Example image 2
figure_name_ex2 = os.path.join(folderInput, "example 2.jpeg")
I_ex2 = cv2.imread(figure_name_ex2, cv2.IMREAD_UNCHANGED)
I_ex2 = cv2.cvtColor(I_ex2, cv2.COLOR_BGR2RGB)
I_ex2 = cv2.resize(I_ex2, (400, 400))

# Read mask image for example 1
mask_img_ex1 = os.path.join(folderInput, "example 1_mask.jpg")
mask_img_1 = cv2.imread(mask_img_ex1, cv2.IMREAD_UNCHANGED)
mask_img_1 = cv2.resize(mask_img_1, (400, 400))

# Read mask image for example 2
mask_img_ex2 = os.path.join(folderInput, "example 2_mask.jpg")
mask_img_2 = cv2.imread(mask_img_ex2, cv2.IMREAD_UNCHANGED)
mask_img_2 = cv2.resize(mask_img_2, (400, 400))

# Image 6
figure_name = 'image6'
figure_name_final=os.path.join(folderInput,figure_name+'_toRestore.tif')
print(figure_name_final)
image_6 = cv2.imread(figure_name_final,cv2.IMREAD_UNCHANGED)

# read mask image
mask_img_name=os.path.join(folderInput,figure_name+'_mask.tif')
mask_img_6 = cv2.imread(mask_img_name,cv2.IMREAD_UNCHANGED)

image_inpainting(I_ex1, mask_img_1, result_folder, 'example_1')
image_inpainting(I_ex2, mask_img_2, result_folder, 'example_2')
image_inpainting(src_image , mask_3_channel , result_folder, 'hola_carola')
image_inpainting(image_6, mask_img_6,result_folder , 'image_6')



