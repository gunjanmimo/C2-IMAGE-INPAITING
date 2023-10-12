from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import cv2
import matplotlib.pyplot as plt


@dataclass
class Parameters:
    hi: float
    hj: float
    dt: float
    iterMax: float
    tol: float


# Creation of the sparse matrix
def sparse(i, j, v, m, n):
    return csr_matrix((v, (i, j)), shape=(m, n))


def sol_Laplace_Equation_Axb(f, dom2Inp, param):
    """_summary_

    Args:
        f (np.array): input image
        dom2Inp (np.array): input mask image, where 0 pixel indicates area to be inpainted
        param (Parameters): parameters which is not required for week 1

    Returns:
        np.array: final inpainted image
    """
    # this code is not intended to be  efficient.

    # input image shape
    ni = f.shape[0]  # height
    nj = f.shape[1]  # width

    # We add the ghost boundaries (for the boundary conditions)

    f_ext = np.zeros((ni + 2, nj + 2), dtype=np.float32)
    ni_ext = f_ext.shape[0]  # height
    nj_ext = f_ext.shape[1]  # width

    # place the original image into the center of the padded (with ghost boundaries) image.
    f_ext[1 : (ni_ext - 1), 1 : (nj_ext - 1)] = f

    # Initializing a zero-filled array with added ghost boundaries to represent the extended mask.
    dom2Inp_ext = np.zeros((ni + 2, nj + 2), dtype=np.float32)
    ndi_ext = dom2Inp_ext.shape[0]
    ndj_ext = dom2Inp_ext.shape[1]
    # placing the original mask into the padded zero-filled array image
    dom2Inp_ext[1 : ndi_ext - 1, 1 : ndj_ext - 1] = dom2Inp

    # Store memory for the A matrix and the b vector
    nPixels = (ni + 2) * (nj + 2)  # Number of pixels

    # We will create A sparse, this is the number of nonzero positions

    # idx_Ai: Vector for the nonZero i index of matrix A
    # idx_Aj: Vector for the nonZero j index of matrix A
    # a_ij: Vector for the value at position ij of matrix A

    b = np.zeros((nPixels, 1), dtype=np.float32)

    # Vector counter
    idx = 0

    # List to store row indices of the non-zero elements in the sparse matrix A
    idx_Ai = []
    # List to store column indices of the non-zero elements in the sparse matrix A
    idx_Aj = []
    # List to store values of the non-zero elements in the sparse matrix A
    a_ij = []

    # North side boundary conditions
    i = 1  # topmost row of the image (excluding the corner)

    for j in range(1, nj + 3, 1):
        # from image matrix (i, j) coordinates to vectorial(p) coordinate to translates the 2D position to a 1D position
        p = (j - 1) * (ni + 2) + i

        # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b

        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p)
        # we are adding 1 in matrix A for boundary pixel
        a_ij.insert(idx, 1)
        idx = idx + 1

        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p + 1)
        # we are adding -1 in matrix A for immediate next pixel which is part of original image
        a_ij.insert(idx, -1)
        idx = idx + 1

        # for all the boundary condition, we will add 0 in b
        b[p - 1] = 0

    # South side boundary conditions
    # This sets i to the last row of the padded image (with ghost boundary), which corresponds to the south boundary
    i = ni + 2

    # we are iterating over each column j of the padded image, covering the entirety of the south boundary
    for j in range(1, nj + 3, 1):
        p = (j - 1) * (ni + 2) + i

        # u(i,j) is influenced by u(i,j) own and above one u(i−1,j)

        # so on for row, we can say both are one column
        idx_Ai.extend([p, p])
        idx_Aj.extend([p, p - 1])
        # 1 for u(i,j) own and -1 for u(i−1,j) because u(i,j)-u(i−1,j)=0 (boundary condition)
        a_ij.extend([1, -1])
        b[p - 1] = 0

    # West side boundary conditions
    j = 1
    for i in range(1, ni + 3, 1):
        p = (j - 1) * (ni + 2) + i

        idx_Ai.extend([p, p])
        idx_Aj.extend([p, p + ni + 2])
        a_ij.extend([1, -1])
        b[p - 1] = 0

    # East side boundary conditions
    j = nj + 2
    for i in range(1, ni + 3, 1):
        p = (j - 1) * (ni + 2) + i

        idx_Ai.extend([p, p])
        idx_Aj.extend([p, p - ni - 2])
        a_ij.extend([1, -1])
        b[p - 1] = 0

    # Inner points
    ### we are iterating over image pixel values(excluding ghost boundary)
    for j in range(2, nj + 2, 1):
        for i in range(2, ni + 2, 1):
            p = (j - 1) * (ni + 2) + i
            # checks if the current pixel (i,j) is within B region to be inpainted
            if dom2Inp_ext[i - 1, j - 1] == 1:
                # For the Laplace equation, we can use a five-point stencil
                idx_Ai.extend([p, p, p, p, p])
                # p center pixel
                # p+1 right pixel
                # p-1 left pixel
                # p - ni - 2 top pixel
                # p + ni + 2 bottom pixel
                idx_Aj.extend([p, p + 1, p - 1, p + ni + 2, p - ni - 2])

                # coefficient values of laplacian equation
                a_ij.extend([4, -1, -1, -1, -1])
                # for laplacian eqn, we have 0 on b
                b[p - 1] = 0
            else:
                # for pixel in A region
                idx_Ai.append(p)
                idx_Aj.append(p)

                # for pixel in A region, we add 1 in matrix A
                a_ij.append(1)
                # we are exact pixel value of A in b
                b[p - 1] = f_ext[i - 1, j - 1]

    # to start the array at 0
    idx_Ai_c = [i - 1 for i in idx_Ai]
    idx_Aj_c = [i - 1 for i in idx_Aj]

    # A is a sparse matrix, so  for memory requirements we create a sparse matrix
    # TO COMPLETE 7

    """ 
    we calculated values of A matrix, and those values' index stored in idx_Ai_c & idx_Aj_c list, so we can create a sparse matrix 
    of size (nPixels,nPixels)
    """
    A = sparse(
        idx_Ai_c, idx_Aj_c, a_ij, nPixels, nPixels
    )  # ??? and ???? is the size of matrix A
    # Solve the sistem of equations
    # solve this x of Ax=b linear equation
    x = spsolve(A, b)

    # From vector to matrix
    # reshape the 1D vector to a 2D matrix
    u_ext = np.reshape(x, (ni + 2, nj + 2), order="F")
    u_ext_i = u_ext.shape[0]
    u_ext_j = u_ext.shape[1]

    # Eliminate the ghost boundaries and extract the final image without ghost boundary
    u = np.full((ni, nj), u_ext[1 : u_ext_i - 1, 1 : u_ext_j - 1], order="F")
    return u


