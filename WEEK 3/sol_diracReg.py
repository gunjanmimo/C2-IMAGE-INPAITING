import math
import numpy as np


def sol_diracReg(x, epsilon):
    # Dirac function of x
    # sol_diracReg(x, epsilon) Computes the derivative of the heaviside
    # function of x with respect to x.Regularized based on epsilon.

    y = epsilon / (math.pi * (epsilon**2 + x**2))  # TO DO 19: Line to complete

    return y
