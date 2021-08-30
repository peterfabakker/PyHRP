import pandas as pd
import numpy as np
from math import fabs
from ..pyhrp.tools.weightallocation import ClusteringAllocation

def test_get_all_leafs():
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

    Return all leafs of cluster 5: [2, 3, 0]
    """
    link_matrix = np.array([
            [2, 3, 0.1, 2],
            [4, 0, 0.2, 3],
            [5, 1, 0.3, 4],
            ])
    cluster_num = 5
    clustering_allocation = ClusteringAllocation(quasi_diag_cov=None, tickers=None, link_matrix=link_matrix, strategy='ivp', returns=None)
    assert clustering_allocation.get_all_leafs(cluster_num) == [2,3,0]

def test_ERC_min_function():
    var_cov = np.array([[1, 1],
                        [1, 1]])
    weights = np.array([2/3,1/3])
    clustering_allocation = ClusteringAllocation(quasi_diag_cov=None, tickers=None, link_matrix=None, strategy='ivp', returns=None)
    assert fabs(clustering_allocation.ERC_min_function(weights, var_cov) - 2/9)  < 0.001

def test_jac_ERC_min_function():
    var_cov = np.array([[1, 1],
                        [1, 1]])
    weights = np.array([3.9/4,0.1/4])
    clustering_allocation = ClusteringAllocation(quasi_diag_cov=None, tickers=None, link_matrix=None, strategy='erc', returns=None)
    jac = clustering_allocation.jac_ERC_min_function(weights, var_cov)
    # print(jac)
    # print(len(jac))
    # print(type(jac))

def test_ERC_optimization():
    var_cov = np.array([[4, 0, 0],
                        [0, 16, 0],
                        [0, 0, 25]])
    clustering_allocation = ClusteringAllocation(quasi_diag_cov=None, tickers=None, link_matrix=None, strategy='erc', returns=None)
    w = clustering_allocation.ERC_optimization(var_cov)
    assert fabs(w[0] - (1/2)/(1/2 + 1/4 + 1/5)) < 0.001
    assert fabs(w[1] - (1/4)/(1/2 + 1/4 + 1/5)) < 0.001
    assert fabs(w[2] - (1/5)/(1/2 + 1/4 + 1/5)) < 0.001

def test_ERC_optimization2():
    var_cov = np.array([[4, -0.5, 0.2],
                        [-0.5, 16, 0.5],
                        [0.2, 0.5, 25]])
    clustering_allocation = ClusteringAllocation(quasi_diag_cov=None, tickers=None, link_matrix=None, strategy='erc', returns=None)
    w = clustering_allocation.ERC_optimization(var_cov)
    for i in range(3):
        for j in range(3):
            assert fabs(w[i]*(var_cov @ w)[i] - w[j]*(var_cov @ w)[j]) < 0.001

