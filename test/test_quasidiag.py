import pandas as pd
import numpy as np
from math import fabs
from ..pyhrp.tools.quasidiag import quasidiagonalize_cov

def test_4by4_case():
    """
    Given a link matrix with the below representation:
           ______|______
       ____|_____     |
    ___|___     |     |
    |     |     |     |
    C     D     A     B
    2     3     0     1
    
    (C,D) ---> 4
    (C, D, A) ---> 5
    (C, D, A, B) ---> 6

    Return a covariance matrix with columns & rows in order C D A B. 
    """
    d = {'A': [1, 2, 3], 'B': [3, 2, 1], 'C': [4, 5, 6], 'D': [6, 5, 4]}
    returns = pd.DataFrame(data=d)
    link_matrix = [
        [2, 3, 0.1, 2],
        [4, 0, 0.2, 3],
        [5, 1, 0.3, 4],
        ]

    quasi_diag_cov = quasidiagonalize_cov(returns, link_matrix)

    c = {'D': [6, 5, 4], 'C': [4, 5, 6], 'A': [1, 2, 3], 'B': [3, 2, 1]}
    true_cov = pd.DataFrame(data=c).cov()
    m = true_cov.shape[0]

    for i in range(0, m):
        for j in range(i+1, m):
            assert fabs(quasi_diag_cov.iloc[i, j] - true_cov.iloc[i, j]) < 1e-5

