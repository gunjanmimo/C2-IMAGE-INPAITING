def sol_DjFwd(I, hj ):
    # Compute the Forward finite differences with respect to the
    # j coordinate only for the 1:end-1 columns. The last column is not replaced

    ni = I.shape[0]
    nj = I.shape[1]

    if 'hj' not in locals():
        hj = 1

    result = I

    result[:, :nj - 1] = (I[:, 1:nj] - I[:, :nj - 1]) / hj


    return result