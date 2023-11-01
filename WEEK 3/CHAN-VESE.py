# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

# Load the image in grayscale
image = cv2.imread("./Screenshot 2023-10-29 at 00.42.02.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(image)
plt.show()

rows, cols = image.shape

# Define initial contour as a circle in the center
center = (int(rows / 2), int(cols / 2))
radius = 200  # arbitrary value, can be adjusted


def initialize_phi():
    phi = np.ones((rows, cols)) * -1
    return phi


phi = initialize_phi()
cv2.circle(phi, center, radius, 1, -1)

plt.imshow(phi)
plt.show()

lambda1 = 1.0
lambda2 = 1.0
mu = 0.1
nu = 0
tau = 0.1
num_iterations = 1000
epsilon = 1e-3  # to prevent division by zero when computing delta function


def dirac_delta(x, epsilon):
    return (1.0 / np.pi) * (epsilon / (x**2 + epsilon**2))


def curvature(phi):
    fy, fx = np.gradient(phi)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)

    # calculating divergence
    Nyy, Nxx = np.gradient(Ny)
    Nxy, Nyx = np.gradient(Nx)
    div = Nxx + Nyy

    return div


stopping_threshold = 0.25
for i in tqdm(range(num_iterations)):
    old_phi = phi.copy()
    if i % 50 == 0:
        phi = initialize_phi()
    # update c1 and c2
    c1 = np.sum(image[phi > 0]) / (np.sum(phi > 0) + 1e-8)
    c2 = np.sum(image[phi <= 0]) / (np.sum(phi <= 0) + 1e-8)

    delta_phi = dirac_delta(phi, epsilon)
    F = lambda1 * (image - c1) ** 2 - lambda2 * (image - c2) ** 2
    div = curvature(phi)

    # update phi
    phi = phi + tau * (mu * div - nu - F * delta_phi)

    # stopping criteria
    if np.max(np.abs(phi - old_phi)) < stopping_threshold:
        break


contour = np.uint8(phi > 0) * 255
plt.imshow(contour, cmap="gray")
plt.show()

# %%


# FARMACIA JOSE SANCHEZ, Plaça de Sant Agustí Vell, 11, 08003 Barcelona
