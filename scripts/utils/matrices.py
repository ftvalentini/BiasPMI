import numpy as np

from scipy import sparse


def is_symmetric(M, verbose=True):
    """
    Test symmetry of sparse matrix
    Param:
        - verbose: prints values and indices of conflicting matrix entries
    """
    diffs_matrix = abs(M - M.T)
    diffs_matrix.data = np.round(diffs_matrix.data, 8) # 8 decimales tolerance
    diffs_matrix.eliminate_zeros() # sets 0 as sparse 0
    if diffs_matrix.nnz == 0:
        return True
    ii_nz, jj_nz = (diffs_matrix).nonzero() # nonzero row/col indices
    if verbose:
        print("Conflicting entries:")
        for pair in zip(ii_nz, jj_nz):
            print(f"M[{pair}] = {M[pair]}")
    return False
