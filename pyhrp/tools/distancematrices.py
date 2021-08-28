"""
The distance matrices module. This module provides various types of distance matrices that can be used in hierarchical clustering.
"""
import pandas as pd
import numpy as np
import math
from abc import ABC, abstractmethod
from scipy.optimize import minimize, Bounds

class DistanceMatrix(ABC):
    """ An abstract class for classes that represent distance matrices.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_distance_matrix(self):
        """
        Returns some type of NxN matrix of distances.

        Returns
        -------
        NxN pandas dataframe
        """
        pass

    def get_compressed_distance_matrix(self):
        """
        Returns a 1-D ndarray of size N-choose-2. The vector returned is simply a compressed 
        version of the matrix returned in the get_distance_matrix method.

        Returns
        -------
        1-D ndarray of size N-choose-2
        """
        matrix = self.get_distance_matrix()
        m = matrix.shape[0]
        compressed_vector = np.zeros(int(math.factorial(m) / (math.factorial(2) * math.factorial(m-2))))

        for i in range(0, m):
            for j in range(i+1, m):
                compressed_vector[int(m*i + j - (((i+2)*(i+1))/2))] = matrix.iloc[i,j]
    
        return compressed_vector

class CorrDistance(DistanceMatrix):
    """ 
    Given a NxN matrix of asset returns correlations, creates an NxN matrix of distances between assets
    using the formula D_(i,j) = sqrt(1/2 * (1 - r_(i,j))), where r_(i,j) is the correlation
    between assets i and j.

    """
    def __init__(self, corr):
        """
        Parameters
        ----------
        corr: NxN pandas dataframe
        """
        self.corr = corr

    def get_distance_matrix(self):
        """
        Overrides DistanceMatrixInterface.get_distance_matrix. Returns an (NxN) dataframe of distances.

        Returns
        -------
        NxN pandas dataframe
        """
        shape = self.corr.shape
        if shape[0] != shape[1]:
            print('ERROR: not a square matrix.')

        N = shape[0]
        distance_matrix = self.corr.copy()
    
        for i in range(N):
            for j in range(N):
                distance_matrix.iloc[i,j] = math.sqrt(0.5 * (1 - self.corr.iloc[i,j]))

        return(distance_matrix)

    def get_compressed_distance_matrix(self):
        """
        Returns a compressed version of the matrix returned by the get_distance_matrix method.

        Returns
        -------
        1-D ndarray of size N-choose-2
        """
        return super().get_compressed_distance_matrix()

class PortfolioDistance(DistanceMatrix):
    """
    Given a NxN matrix of correlations, creates an NxN matrix of distances between assets
    using the formula K_{i,j} = sqrt((sum_k(D_{k,i} - D_{k,j})^2), where D_{i,j} is the distance between 
    asset i and j as defined by CorrDistance.
    """
    def __init__(self, corr):
        """
        Parameters
        ----------
        corr: NxN pandas dataframe
        """
        self.zoc_obj = CorrDistance(corr)
        self.corr = corr

    def get_distance_matrix(self):
        """
        Overrides DistanceMatrixInterface.get_distance_matrix. Returns an (NxN) dataframe of portfolio distances.

        Returns
        -------
        NxN pandas dataframe
        """
        zoc = self.zoc_obj.get_distance_matrix()
        shape = zoc.shape
        if shape[0] != shape[1]:
            print('ERROR: not a square matrix.')

        N = shape[0]
        distance_matrix = zoc.copy()
    
        for i in range(N):
            for j in range(N):
            
                distance_matrix.iloc[i,j] = np.linalg.norm(np.matrix(zoc.iloc[:,i]) - np.matrix(zoc.iloc[:,j]))

        return(distance_matrix)

    def get_compressed_distance_matrix(self):
        """
        Returns a compressed version of the matrix returned by the get_distance_matrix method.

        Returns
        -------
        1-D ndarray of size N-choose-2
        """
        return super().get_compressed_distance_matrix()

class LTDCDistance(DistanceMatrix):
    """
    A class for constructing an NxN matrix, where element (i,j) is the lower tail dependency
    coeffecient (LTDC) between timeseries i and j.
    """
    def __init__(self, timeseries, copula):
        """
        Parameters
        ----------
        timeseries: MxN pandas dataframe
            Each column represents a timeseries of length N. It is important
            to preprocess the data so that each timeseries is approximately iid. One may of doing this is by 
            applying an AR(1)-GARCH(1) model on each timeseries seperately, and substituting in the standardized 
            residuals of the model. The correlation structure between securities will remain intact, but
            most of the timeseries will be closer to iid.

        copula: str
            The copula used to derive the LTDC between pairs of timeseries. As of now, only "clayton" is supported.
        """
        self.timeseries = timeseries
        self.copula = copula

    def get_distance_matrix(self):
        """
        Overrides DistanceMatrixInterface.get_distance_matrix. Returns an (NxN) dataframe of dissimilarities
        equal to the negative log-likelihood of the LTDC between any two timeseries.

        Returns
        -------
        NxN pandas dataframe
        """
        N = self.timeseries.shape[1]
        dist = pd.DataFrame(np.zeros((N, N)))
        for i in range(N):
            for j in range(i, N):
                ltdc = self.get_LTDC_MLE(i,j)
                epsilon = 1e-6
                dist.iloc[i,j] = -math.log(ltdc + epsilon) 
        
        for i in range(N):
            for j in range(0, i):
                dist.iloc[i,j] = dist.iloc[j, i]

        return dist

    def get_LTDC_MLE(self, i, j):
        """
        Estimate the maximum likelihood estimate of the LTDC between timeseries i and j
        using ``self.copula`` copula.
        
        Parameters
        ----------
        i: int
        j: int

        Returns
        -------
        float in [0, 1]
        """
        if self.copula == "clayton":
            ltdc = self.get_LTDC_MLE_clayton(i, j)
        
        return ltdc

    def get_LTDC_MLE_clayton(self, i, j):
        """
        Uses gradient descent to find the maximum likelihood estimate of the LTDC 
        between timeseries i and j using the Clayton copula. Gradient clipping is 
        used to maintain stability during optimization. Step size is scaled by .95
        at each step. Note that this algorithm forces ``theta`` to be greater than 0.
        
        Parameters
        ----------
        i: int
        j: int

        Returns
        -------
        float in [0, 1]
        """
        bounds = Bounds(1e-5, 10)
        theta0 = [1]
        def neg_clayton_log_likelihood(theta):
            return -self.clayton_log_likelihood(theta, i, j)

        def neg_clayton_d_log_likelihood(theta):
            return -self.clayton_d_log_likelihood(theta, i, j)

        opt = minimize(neg_clayton_log_likelihood, theta0, method='SLSQP', 
                        jac=neg_clayton_d_log_likelihood, 
                        options={'ftol':1e-9, 'disp': True},
                        bounds=bounds)
        theta = opt.x
        
        return (1/2)**(1/theta)

    def clayton_d_log_likelihood(self, theta, i, j):
        """
        The first derivative of the log-likelihood function for the Clayton copula evaluated
        at ``theta``.

        Parameters
        ----------
        i: int
        j: int
        theta: float in (-1, inf)\{0}

        Returns
        -------
        float
        """
        series_i = self.timeseries.iloc[:, i]
        series_j = self.timeseries.iloc[:, j]
        n = len(series_i)
        d_log_likelihood = np.zeros(n)

        for p in range(n):
            F_i = self.get_empirical_CDF(series_i, series_i[p])
            F_j = self.get_empirical_CDF(series_j, series_j[p])
            g = F_i**(-theta) + F_j**(-theta) - 1
            d_g = (-F_i**(-theta))*math.log(F_i) - (F_j**(-theta))*math.log(F_j)

            if (g)**(-1/theta) > 0:
                d_log_likelihood[p] = 1/(1+theta) + 1/(theta**2)*math.log(g) - ((1/theta)+2)*d_g/g - math.log(F_i*F_j)
            else:
                print('WARNING: Derivative undefined')
        
        return np.sum(d_log_likelihood)


    def clayton_log_likelihood(self, theta, i, j):
        """
        The log-likelihood function for the Clayton copula evaluated at ``theta``.

        Parameters
        ----------
        i: int
        j: int
        theta: float in (-1, inf)\{0}

        Returns
        -------
        float
        """
        series_i = self.timeseries.iloc[:, i]
        series_j = self.timeseries.iloc[:, j]
        n = len(series_i)
        log_likelihood = np.zeros(n)
        
        for p in range(n):
            F_i = self.get_empirical_CDF(series_i, series_i[p])
            F_j = self.get_empirical_CDF(series_j, series_j[p])
            g = F_i**-theta + F_j**-theta - 1
            
            if (g)**(-1/theta) > 0:
                log_likelihood[p] = math.log(1+theta) - ((1/theta)+2)*math.log(g) - (theta+1)*math.log(F_i*F_j)
            else:
                log_likelihood[p] = -1e10
        return np.sum(log_likelihood)


    def get_empirical_CDF(self, sample, quantile):
        """
        Parameters
        ----------
        sample: ndarray
        quantile: float

        Returns
        -------
        float in [0, 1]
            The empirical probability of sampling a value less than ``quantile``
            from the distribution that generated the values in ``sample``. 
        """
        n = len(sample)
        sum = 0
        for i in range(n):
            sum += (sample[i] <= quantile)

        return sum/(n+1)

    def get_compressed_distance_matrix(self):
        return super().get_compressed_distance_matrix()
