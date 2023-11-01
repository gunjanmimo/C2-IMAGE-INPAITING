
import cv2
import numpy as np
import sol_chanvese_test
import matplotlib.pyplot as plt

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

figure_path = "/Users/dianatat/Documents/Master/C2 Optimisation techniques for CV/Project/week3/to_complete/Image_to_Restore.png"

I1 = cv2.imread(figure_path, cv2.IMREAD_UNCHANGED)

I=I1.astype('float')

if len(I1.shape)>2:
    I = np.mean(I, axis=2)
    
print(I.shape)

min_val = np.min(I.ravel())
max_val = np.max(I.ravel())
I = (I.astype('float') - min_val)
I = I/max_val

# height, width, number of channels in image
height_mask = I.shape[0]
width_mask = I.shape[1]
dimensions_mask = I.shape

ni=height_mask
nj=width_mask

#Lenght and area parameters
#circles.png mu=1, mu=2, mu=10
#noisedCircles.tif mu=0.1
#phantom17 mu=1, mu=2, mu=10
#phantom18 mu=0.2 mu=0.5
#hola carola
mu=1
nu=0

#Parameters
#lambda1=1;
#lambda2=1;
lambda1 = 1e-3  # Hola carola problem
lambda2 = 1e-3  # Hola carola problem


epHeaviside=1
eta=0.01
#eta=1

tol=0.1
#dt=(10^-2)/mu;
dt=(10^-1)/mu
iterMax=100000
#reIni=0; %Try both of them
#reIni=500;
reIni=100

X, Y = np.meshgrid(np.arange(0,nj), np.arange(0,ni),indexing='xy')

#Initial phi
#phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/2)).^2)+50);

# This initialization allows a faster convergence for phantom 18
#phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/4)).^2)+50);
#Normalization of the initial phi to [-1 1]
#phi_0=phi_0-min(phi_0(:));
#phi_0=2*phi_0/max(phi_0(:));
#phi_0=phi_0-1;

phi_0=I #For the Hola carola problem

min_val = np.min(phi_0)
max_val = np.max(phi_0)

phi_0=phi_0-min_val
phi_0=2*phi_0/max_val
phi_0=phi_0-1

#Explicit Gradient Descent
seg=sol_chanvese_test.sol_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni )

# Visualize the input image
plt.figure()
plt.imshow(I, cmap='gray')
plt.title('Input Image')
plt.axis('off')
plt.show()

# Visualize the segmented image
plt.figure()
plt.imshow(seg, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')
plt.show()

