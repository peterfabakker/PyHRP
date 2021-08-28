"""
The quasi-diagonalization module. Used for creating a quasi-diagonalized covariance matrix, where similar securities
are placed close to one another so that the largest covariances (in absolute value) occur around the diagonal.  
"""
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

def quasidiagonalize_cov(returns, link_matrix):
    """
    Quasi-Diagonalizes the covariance matrix of the NxT returns matrix.

    Parameters
    ----------
    returns: NxT pandas dataframe
             Returns of N securities in our portfolio over T time periods.
    link_matrix: (N-1)x4 ndarray
                 The linkage matrix output by linkage() function in scipy.cluster.hierarchy 
    
    Returns
    -------
    NxN pandas dataframe
    """
    labels = list(returns.columns.values)
    dend = dendrogram(
        link_matrix,
        orientation='top',
        labels=labels,
        distance_sort='descending',
        show_leaf_counts=True
        )
    serialized_columns = dend['ivl']

    return returns[serialized_columns].cov()

def quasidiagonalize_corr(returns, link_matrix):
    """
    Quasi-Diagonalizes the correlation matrix of the NxT returns matrix.

    Parameters
    ----------
    returns: NxT pandas dataframe
             Returns of N securities in our portfolio over T time periods.
    link_matrix: (N-1)x4 ndarray
                 The linkage matrix output by linkage() function in scipy.cluster.hierarchy 
    
    Returns
    -------
    NxN pandas dataframe
    """
    labels = list(returns.columns.values) #numpy.ndarrays don't return names of columns in an ordered list, so this is a shortcut for the time being.
    dend = dendrogram(
        link_matrix,
        orientation='top',
        labels=labels,
        distance_sort='descending',
        show_leaf_counts=True
        )
    serialized_columns = dend['ivl']
   
    return returns[serialized_columns].corr()

    
    

    