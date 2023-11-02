
import numpy as np
import matplotlib.pyplot as plt
import sol_ChanVeseIpol_GDExp
import cv2  

from dataclasses import dataclass

@dataclass
class Parameters:
    hi: float
    hj: float
    dt: float
    mu: float
    nu: float
    iterMax: float
    tol: float

# ======>>>>  input data  <<<<=======

def load_and_preprocess_image(image_path):
    I1 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    I = I1.astype('float')

    if len(I1.shape) > 2:
        I = np.mean(I, axis=2)

    min_val = np.min(I.ravel())
    max_val = np.max(I.ravel())
    I = (I.astype('float') - min_val)
    I = I / max_val

    return I

# Load and preprocess the image you want to segment
image_path = "/Users/dianatat/Downloads/IMG_2748.png"
I = load_and_preprocess_image(image_path)

ni, nj = I.shape

# Set segmentation parameters
mu = 0.2
nu = 0
lambda1 = 1
lambda2 = 1
epHeaviside = 1
eta = 0.01
tol = 0.1
dt = (10^-1) / mu
iterMax = 100
reIni = 100

# Define the new center coordinates to move the circle
new_x = 100  
new_y = 150  

phi_0 = np.ones((ni, nj)) * -1

# Update the center to move the circle
center = (new_x, new_y)

# Specify the radius and draw the circle
radius = min(ni, nj) // 2  
cv2.circle(phi_0, center, radius, 1, -1)

min_val = np.min(phi_0)
max_val = np.max(phi_0)

phi_0 = 2 * phi_0 / max_val
phi_0 = phi_0 - 1

seg = sol_ChanVeseIpol_GDExp.sol_ChanVeseIpol_GDExp(
    I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni
)
seg = np.uint8(seg > 0) * 255

plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Input Image')
plt.axis('off')

# Subplot 2: Segmented Image
plt.subplot(1, 2, 2)
plt.imshow(seg, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

plt.show()
