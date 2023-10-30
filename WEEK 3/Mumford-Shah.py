# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
# PARAMETERS
alpha = 0.01
beta = 0.01
epsilon = 0.1
num_iterations = 500


def laplacian(img):
    gx, gy = np.gradient(img)
    return np.gradient(gx)[0] + np.gradient(gy)[1]


def update_v(u, epsilon, alpha, beta):
    laplacian_u = laplacian(u)
    numerator = epsilon * laplacian_u + 1 / (4 * epsilon)
    grad_u = np.linalg.norm(np.gradient(u), axis=0)
    denominator = (
        numerator + alpha * grad_u**2 + beta + 1e-8
    )  # Add a small constant for regularization
    v_new = numerator / denominator
    return v_new


def update_u(u, u0, v, alpha):
    grad_u = np.array(np.gradient(u))
    divergence_v2_grad_u = (
        np.gradient(v**2 * grad_u[0])[0] + np.gradient(v**2 * grad_u[1])[1]
    )
    u_new = (u0 + alpha * divergence_v2_grad_u) / (1 + alpha * v**2)
    return u_new


def optimize(u0: np.array, u: np.array, v: np.array, num_iter: int = num_iterations):
    assert len(u.shape) == 2, f"IMAGE SHAPE SHOULD BE 2D, BUT GOT {len(u.shape)}D"
    for _ in tqdm(range(num_iter)):
        u = update_u(u=u, u0=u0, v=v, alpha=alpha)
        v = update_v(u=u, epsilon=epsilon, alpha=alpha, beta=beta)
    return u


if __name__ == "__main__":
    image_path = "./Screenshot 2023-10-29 at 00.42.02.png"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0  # Normalize to range [0, 1]
    # v = cv2.medianBlur(img, 3)
    # v = np.random.randint(low=0, high=255, size=img.shape)
    v = np.ones_like(img) * 0.5
    final_output = optimize(u0=img, u=img, v=v, num_iter=num_iterations)

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(final_output, cmap="gray")
    plt.axis("off")
    plt.show()


# %%
