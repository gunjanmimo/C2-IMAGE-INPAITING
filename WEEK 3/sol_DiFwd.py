import numpy as np

def sol_DiFwd(I, hi ):
    #Compute the Forward finite differences with respect to the
    #i coordinate only for the 1:end-1 rows. The last row is not replaced

    ni=I.shape[0]
    nj=I.shape[1]

    if 'hi' not in locals():
        hi = 1

    result = I
    # Begin To Complete 8
    result[:ni - 1, :] = (I[1:ni, :] - I[:ni - 1, :]) / hi
    #End To Complete 8

    return result