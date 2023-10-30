def sol_DjBwd( I, hj):
    # Compute the backward finite differences with respect to the
    # j coordinate only for the 2: end columns.The first column is not replaced

    ni = I.shape[0]
    nj = I.shape[1]

    if 'hj' not in locals():
        hj = 1

    result = I

    result[:, 1:nj] = (I[:, 1:nj] - I[:, 0:nj - 1]) / hj


    return result