import math
import numpy as np
from scipy import ndimage

# local imports
import sol_diracReg


def sol_ChanVeseIpol_GDExp(
    I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni
):
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

        I1f = I1.astype("float")
        I0f = I0.astype("float")

        # Minimization w.r.t c1 and c2(constant estimation)

        c1 = np.mean(I1f)  # ! Average intensity inside the contour # TODO 1: Line to complete
        c2 = np.mean(I0f)  # ! Average intensity outside the contour # TODO 2: Line to complete

        # Boundary conditions
        phi_f[0, :] = phi_f[1, :]  # ! Top boundary #TODO 3: Line to complete
        phi_f[ni - 1, :] = phi_f[ni - 2, :]  # ! Bottom boundary #TODO 4: Line to complete

        phi_f[:, 0] = phi_f[:, 1]  # ! Left boundary #TODO 5: Line to complete
        phi_f[:, nj - 1] = phi_f[:, nj - 2]  # ! Right boundary #TODO 6: Line to complete

        # phi_f variable is mutable along the function. A trick to avoid this:
        phi_c = phi_f[:]

        new_phi2 = np.zeros(phi_c.shape, dtype=np.float32)
        new_phi3 = np.zeros(phi_c.shape, dtype=np.float32)
        new_phi4 = np.zeros(phi_c.shape, dtype=np.float32)
        new_phi = np.zeros(phi_c.shape, dtype=np.float32)
        phi = np.zeros(phi_c.shape, dtype=np.float32)

        for i in range(0, phi_c.shape[0]):
            for j in range(0, phi_c.shape[1]):
                new_phi[i, j] = phi_c[i, j]
                new_phi2[i, j] = phi_c[i, j]
                new_phi3[i, j] = phi_c[i, j]
                new_phi4[i, j] = phi_c[i, j]
                phi[i, j] = phi_c[i, j]
        # Regularized Dirac 's Delta computation
        delta_phi = sol_diracReg.sol_diracReg(phi_c, epHeaviside)  # H'(phi)

        # i direction (vertical) #TODO 7: Line to complete
        phi_iFwd = (
            np.roll(phi_f, -1, axis=0) - phi_f
        )  # Forward difference in i direction #TODO 8: Line to complete
        phi_iBwd = phi_f - np.roll( 
            phi_f, 1, axis=0
        )  # Backward difference in i direction

        # j direction (horizontal) #TODO 9: Line to complete
        phi_jFwd = (
            np.roll(phi_f, -1, axis=1) - phi_f
        )  # Forward difference in j direction #TODO 10: Line to complete
        phi_jBwd = phi_f - np.roll(
            phi_f, 1, axis=1
        )  # Backward difference in j direction

        # Centered differences #TODO 11: Line to complete
        phi_icent = ( 
            np.roll(phi_f, -1, axis=0) - np.roll(phi_f, 1, axis=0)
        ) / 2.0  # Centered difference in i direction #TODO 12: Line to complete
        phi_jcent = (
            np.roll(phi_f, -1, axis=1) - np.roll(phi_f, 1, axis=1)
        ) / 2.0  # Centered difference in j direction

        # A and B estimation (A y B from the Pascal Getreuer's IPOL paper "Chan Vese segmentation
        A = delta_phi * (I - c1) ** 2  # Estimation of A #TODO 13: Line to complete
        B = delta_phi * (I - c2) ** 2  # Estimation of B #TODO 14: Line to complete

        # Equation 22, for inner points
        # Calculate Laplacian of phi
        laplacian_phi = (
            np.roll(phi_f, -1, axis=0) + np.roll(phi_f, 1, axis=0) - 2 * phi_f
        ) / (hi**2) + (
            np.roll(phi_f, -1, axis=1) + np.roll(phi_f, 1, axis=1) - 2 * phi_f
        ) / (
            hj**2
        )

        # Update new_phi4 using Equation 22 #TODO 22: Line to complete
        new_phi4 = phi_f - dt * (
            mu * laplacian_phi - nu + lambda1 * (I - c1) ** 2 - lambda2 * (I - c2) ** 2
        )

        if reIni > 0 & np.mod(nIter, reIni) == 0:
            indGT1 = new_phi4 >= 0
            indGT = indGT1.astype("float")
            indLT1 = new_phi4 < 0
            indLT = indLT1.astype("float")

            xb1 = ndimage.distance_transform_edt(1 - indLT)
            xb2 = ndimage.distance_transform_edt(1 - indGT)

            new_phi5 = xb1 - xb2

            # Normalization[-1,1]
            nor = min(abs(new_phi5.min()), new_phi5.max())

            # Normalize `phi` by dividing it by `nor`
            phi6 = new_phi5 / nor

            # Diference. This stopping criterium has the problem that phi can
            # change, but not the zero level set, that it really is what we are looking for

            dif = np.mean(np.sum((phi.ravel() - phi_old.ravel()) ** 2))
            phi_old = phi6
            phi_f = phi6
        else:
            phi6 = new_phi4

    return phi6
