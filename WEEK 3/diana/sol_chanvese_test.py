import cv2
import numpy as np
import math
import sol_diracReg
import sol_DiFwd
import sol_DiBwd
import sol_DjFwd
import sol_DjBwd
import os
from scipy import ndimage
import matplotlib.pyplot as plt

def sol_ChanVeseIpol_GDExp(I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni):

    #Implementation of the Chan - Vese segmentation following the explicit % gradient descent in the paper of Pascal Getreur "Chan-Vese Segmentation".
    #It is the equation 19 from that paper
    # I: Gray color image to segment
    # phi_0: Initial phi
    # mu: mu length parameter(regularizer term)
    # nu: nu area parameter(regularizer term)
    # eta: epsilon for the total variation regularization
    # lambda1, lambda2: data fidelity parameters
    # tol: tolerance for the sopping criterium
    # epHeaviside: epsilon for the regularized heaviside.
    # dt: time step
    # iterMax: MAximum number of iterations
    # reIni: Iterations for reinitialization. 0 means no reinitializacion

    folderInput = '/Users/angelicaatehortua/Documents/posdoctorado/sueltos/week2/solution/'

    ni = I.shape[0]
    nj = I.shape[1]

    hi = 1
    hj = 1

    phi_f = phi_0[:]
    dif = math.inf
    nIter = 0
    phi_old = phi_f

    while dif > tol and nIter < iterMax:

        nIter = nIter + 1

        I1 = I[phi_f >= 0]
        I0 = I[phi_f < 0]
        I1f = I1.astype('float')
        I0f = I0.astype('float')

        # Minimization w.r.t c1 and c2(constant estimation)

        c1_numerator = np.sum(I * (phi_0 >= 0))  # Numerator for c1
        c1_denominator = np.sum(phi_0 >= 0)  # Denominator for c1

        c2_numerator = np.sum(I * (phi_0 < 0))  # Numerator for c2
        c2_denominator = np.sum(phi_0 < 0)  # Denominator for c2
        c1 = c1_numerator / c1_denominator

        c2 = c2_numerator / c2_denominator


     # TODO 1: Line to complete
         # TODO 2: Line to complete


        # Boundary conditions

        phi_f[0, :] = phi_f[1, :]     #TODO 3: Line to complete
        phi_f[ni-1, :] = phi_f[ni-2, :]     #TODO 4: Line to complete

        phi_f[:, 0] = phi_f[:, 1]        #TODO 5: Line to complete
        phi_f[:, nj-1] = phi_f[:, nj-2]     #TODO 6: Line to completend)

        # phi_f variable is mutable along the function. A trick to avoid this:
        phi_c=phi_f[:]

        new_phi2 = np.zeros(phi_c.shape, dtype=np.float64)
        new_phi3 = np.zeros(phi_c.shape, dtype=np.float64)
        new_phi4 = np.zeros(phi_c.shape, dtype=np.float64)
        new_phi = np.zeros(phi_c.shape, dtype=np.float64)
        phi = np.zeros(phi_c.shape, dtype=np.float64)
        for i in range(0, phi_c.shape[0]):
            for j in range(0, phi_c.shape[1]):
                new_phi[i, j] = phi_c[i, j]
                new_phi2[i, j] = phi_c[i, j]
                new_phi3[i, j] = phi_c[i, j]
                new_phi4[i, j] = phi_c[i, j]
                phi[i, j] = phi_c[i, j]

        # Regularized Dirac 's Delta computation
        delta_phi = sol_diracReg.sol_diracReg(phi_c, epHeaviside) # H'(phi)


        # derivatives estimation
        # i direction
        phi_iFwd = sol_DiFwd.sol_DiFwd(phi_c, hi)    #TODO 7: Line to complete
        phi_iBwd = sol_DiBwd.sol_DiBwd(phi_c, hi)  #TODO 8: Line to complete

        # j direction
        phi_jFwd = sol_DjFwd.sol_DjFwd(phi_c, hj)   #TODO 9: Line to complete
        phi_jBwd = sol_DjBwd.sol_DjBwd(phi_c, hj)   #TODO 10: Line to complete


        # centered
        phi_icent = (phi_iFwd + phi_iBwd) / 2  #TODO 11: Line to complete
        phi_jcent = (phi_jFwd + phi_jBwd) / 2  #TODO 12: Line to complete

        #A and B estimation (A y B from the Pascal Getreuer's IPOL paper "Chan Vese segmentation
        A = mu / np.sqrt(eta**2 + phi_icent**2 + phi_jcent**2)      #TODO 13: Line to complete
        B = mu * (I - c1)**2 - mu * (I - c2)**2     #TODO 14: Line to complete

# Equation 22: Update phi using the provided expression
        new_phi4[1:ni - 1, 1:nj - 1] = phi[1:ni - 1, 1:nj - 1] + dt * delta_phi[1:ni - 1, 1:nj - 1] * (
            A[1:ni - 1, 1:nj - 1] * (phi_iFwd[1:ni - 1, 1:nj - 1] + phi_iBwd[1:ni - 1, 1:nj - 1] + phi_jFwd[1:ni - 1, 1:nj - 1] + phi_jBwd[1:ni - 1, 1:nj - 1])
            - B[1:ni - 1, 1:nj - 1] - nu - lambda1 * (I[1:ni - 1, 1:nj - 1] - c1)**2
            + lambda2 * (I[1:ni - 1, 1:nj - 1] - c2)**2
        ) / (1 + dt * delta_phi[1:ni - 1, 1:nj - 1] * (
            A[1:ni - 1, 1:nj - 1] + A[0:ni - 2, 1:nj - 1] + B[1:ni - 1, 1:nj - 1] + B[1:ni - 1, 0:nj - 2]
        ))

         # TODO 15: Line to complete

        if reIni > 0 and nIter % reIni == 0:
            indGT1 = new_phi4 >= 0
            indGT = indGT1.astype('float')
            indLT1 = new_phi4 < 0
            indLT = indLT1.astype('float')

            xb1 = ndimage.distance_transform_edt(1 - indLT)
            xb2 = ndimage.distance_transform_edt(1 - indGT)

            new_phi5 = (xb1 - xb2)

            # Normalization[-1,1]
            nor = min(abs(new_phi5.min()), new_phi5.max())

            # Normalize `phi` by dividing it by `nor`
            phi6 = new_phi5 / nor

            # Difference. This stopping criterion has the problem that phi can change, but not the zero level set, which is what we are looking for

            dif = np.mean(np.sum((phi6.ravel() - phi_old.ravel())**2))
            phi_old = phi6
            phi_f = phi6
        else:
            phi6 = new_phi4

            
    return phi6





