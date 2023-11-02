import numpy as np

def sol_DiBwd(I, hi):
    #Compute the backward finite differences with respect to the
    #i coordinate only for the 2:end rows. The first row is not replaced

    ni = I.shape[0]
    nj = I.shape[1]

    if 'hi' not in locals():
        hi = 1

    result = I

    result[1:ni, :] = (I[1:ni, :] - I[0:ni - 1, :]) / hi


    return result