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
        #TODO 1 and 2
        c1_numerator = np.sum(I * ((1/2) * (1 + (2/np.pi) * np.arctan(phi_old[1:-1, 1:-1] / epHeaviside))))
        c1_denominator = np.sum((1/2) * (1 + (2/np.pi) * np.arctan(phi_old[1:-1, 1:-1] / epHeaviside)))
        c1 = c1_numerator / c1_denominator

        c2_numerator = np.sum(I * (1 - ((1/2) * (1 + (2/np.pi) * np.arctan(phi_old[1:-1, 1:-1] / epHeaviside)))))
        c2_denominator = np.sum(1 - ((1/2) * (1 + (2/np.pi) * np.arctan(phi_old[1:-1, 1:-1] / epHeaviside))))
        c2 = c2_numerator / c2_denominator

        # Boundary conditions

        phi[0, :] = phi_old[1, :]  #TODO 3: Line to complete
        phi[-1, :] = phi_old[-2, :]  #TODO 4: Line to complete
        phi[:, 0] = phi_old[:, 1] #TODO 5: Line to complete
        phi[:, -1] = phi_old[:, -2] #TODO 6: Line to completend)

        # phi_f variable is mutable along the function. A trick to avoid this:
        phi_c=phi_f[:]

        new_phi2 = np.zeros(phi_c.shape, dtype=np.float)
        new_phi3 = np.zeros(phi_c.shape, dtype=np.float)
        new_phi4 = np.zeros(phi_c.shape, dtype=np.float)
        new_phi = np.zeros(phi_c.shape, dtype=np.float)
        phi = np.zeros(phi_c.shape, dtype=np.float)
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

        phi_iFwd = (phi_old[1:, 1:-1] - phi_old[:-1, 1:-1]) / hi    #TODO 7: Line to complete
        phi_iBwd = (phi_old[1:, 1:-1] - phi_old[:-1, 1:-1]) / hi  #TODO 8: Line to complete

        # j direction
        phi_jFwd = (phi_old[1:-1, 1:] - phi_old[1:-1, :-1]) / hj   #TODO 9: Line to complete
        phi_jBwd = (phi_old[1:-1, 1:] - phi_old[1:-1, :-1]) / hj   #TODO 10: Line to complete


        # centered
        phi_icent = (phi_iFwd + phi_iBwd) / 2  #TODO 11: Line to complete
        phi_jcent = (phi_jFwd + phi_jBwd) / 2  #TODO 12: Line to complete

        #A and B estimation (A y B from the Pascal Getreuer's IPOL paper "Chan Vese segmentation
        A = mu / np.sqrt((eta**2) + (phi_iFwd**2) + (phi_jcent**2))      #TODO 13: Line to complete
        B = mu / np.sqrt((eta**2) + (phi_icent**2) + (phi_jFwd**2))      #TODO 14: Line to complete

        numer1 = phi_old[2:, 1:-1] * A
        numer2 = phi_old[:-2, 1:-1] * A
        numer3 = phi_old[1:-1, 2:] * B
        numer4 = phi_old[1:-1, :-2] * B

        #Equation 22, for inner points
        new_phi4[1:-1, 1:-1] = (phi_old[1:-1, 1:-1] + dt * delta_phi[1:-1, 1:-1] * (numer1 + numer2 + numer3 + numer4 - nu - lambda1 * (I - c1)**2 + lambda2 * (I - c2)**2)) / (1 + dt * delta_phi[1:-1, 1:-1] * (A + A[:-1, 1:-1] + B[1:-1, 1:-1] + B[1:-1, :-1]))
        # TODO 15: Line to complete

        if reIni > 0 and np.mod(nIter, reIni) == 0:

            indGT1 = new_phi4 >= 0
            indGT = indGT1.astype('float')
            indLT1 = new_phi4 < 0
            indLT=indLT1.astype('float')

            xb1 = ndimage.distance_transform_edt(1-indLT)
            xb2 = ndimage.distance_transform_edt(1-indGT)

            new_phi5 = (xb1-xb2)

            # Normalization[-1,1]
            nor = min(abs(new_phi5.min()), new_phi5.max())

            # Normalize `phi` by dividing it by `nor`
            phi6 = new_phi5 / nor

            #Diference. This stopping criterium has the problem that phi can
            #change, but not the zero level set, that it really is what we are looking for

            dif = np.mean(np.sum((phi.ravel() - phi_old.ravel())**2))
            phi_old = phi6
            phi_f=phi6
        else:
            phi6=new_phi4

    return phi6

