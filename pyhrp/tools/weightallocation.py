"""
The weight allocation module. 
"""
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize, Bounds
from abc import ABC, abstractmethod

class WeightAllocation(ABC):
    """
    An abstract class for allocating securities within a portfolio.
    """
    
    @abstractmethod
    def get_portfolio_weights(self):
        """
        Given a quasi-diagonal covariance matrix, returns portfolio holdings of securities.

        Returns
        -------
        dict 
            dictionary entries should be of the form str:float32. 
            i.e. {'AAPL': 0.32, 'XOM': 0.24, ...}
            All allocations are between 0 and 1 (inclusive).
        """

        pass

class ClusteringAllocation(WeightAllocation):
    """
    A class for clustering-based weight allocation.
    """
    def __init__(self, quasi_diag_cov, tickers, link_matrix, strategy, returns):
        """
        Parameters
        ----------
        quasi_diag_cov: pd.Dataframe
            The quasidiagonalized covariance matrix
        tickers: list of strings
            The list of tickers, with tickers ordered identically to the original covariance matrix
        link_matrix: ndarray
            The matrix output by scipy.cluster.hierarchy.linkage
        strategy: str
            The type of clustering strategy that will be used. Can be set to 'minvar', 
            'erc', and 'ivp' for the minimum variance, equal risk contribution, and
            inverse volatility portfolios respectively.
        returns: pd.DataFrame
            A dataframe of timeseries of returns. If using the 'ltdc' dissimilarity measure, must be 
            preprocessed to remove autocorrelation and heteroscedasticity.
        """
        self.quasi_diag_cov = quasi_diag_cov
        self.tickers = tickers
        self.link_matrix = link_matrix
        self.strategy = strategy
        self.returns = returns

    def get_portfolio_weights(self):
        """
        Returns portfolio weights using clustering-based weight allocation.

        Returns
        -------
        dict
            Dictionary with tickers as keys and each ticker's respective weight as values

        """
        weights_init = {idx:1 for idx in range(len(self.tickers))} 
        cluster_num = 2*(len(self.tickers) - 1) 
        weights = self.clustering_allocation(cluster_num, weights_init)

        ticker_holdings = {}
        sum = 0
        for key in weights:
            ticker_holdings[self.tickers[key]] = weights[key]
            sum += weights[key]

        print(ticker_holdings)
        return ticker_holdings

    def clustering_allocation(self, cluster_num, weights):
        """
        Recursive function for determining portfolio weights for assets clustered using agglomerative
        hierarchical clustering.

        Parameters
        ----------
        cluster_num: int
            The integer denoting the current cluster
        weights: dict
            A dictionary containing indices as keys and asset weights as values;
            Each key represents a particular asset

        Returns
        -------
        dict
            Dictionary with tickers as keys and each ticker's respective weight (calculated 
            so far) as values
        """
        if cluster_num < len(self.link_matrix) + 1:
            return weights

        cluster_idx = cluster_num - len(self.tickers) 
        left_cluster_num = int(self.link_matrix[cluster_idx, 0])
        right_cluster_num = int(self.link_matrix[cluster_idx, 1])
        left_leafs = self.get_all_leafs(left_cluster_num) 
        right_leafs = self.get_all_leafs(right_cluster_num)

        left_var_cov = self.get_cluster_cov_matrix(left_leafs)
        right_var_cov = self.get_cluster_cov_matrix(right_leafs)
        
        left_w = self.get_weights(left_var_cov) 
        right_w = self.get_weights(right_var_cov)

        # compute the variance of left & right child, and the covariance between the childs:
        left_variance = left_w.T @ left_var_cov @ left_w
        right_variance = right_w.T @ right_var_cov @ right_w
        cov = np.cov(self.returns[[self.tickers[i] for i in left_leafs]].to_numpy() @ left_w, 
                        self.returns[[self.tickers[i] for i in right_leafs]].to_numpy() @ right_w)

        # compute the weighting factors of left & right child:
        left_alpha, right_alpha = self.get_cluster_factors(left_variance, right_variance, cov[0,1]) 

        left_weights = {}
        right_weights = {}

        for idx in left_leafs:
            left_weights[idx] = weights[idx] * left_alpha

        for idx in right_leafs:
            right_weights[idx] = weights[idx] * right_alpha

        return {**self.clustering_allocation(left_cluster_num, left_weights), 
                **self.clustering_allocation(right_cluster_num, right_weights)}

    def get_cluster_factors(self, left_var, right_var, cov):
        """
        Returns the factors to weight the left cluster and right cluster.

        Parameters
        ----------
        left_var: float
            The variance of the left cluster
        right_var: float
            The variance of the right cluster
        cov: float
            The covariance between the left and right cluster
        Returns
        -------
        float, float
        """
        if self.strategy == "minvar":
            left_alpha = (1/2)*(2*right_var - cov)/(left_var + right_var - cov)
            right_alpha = 1 - left_alpha

        elif  self.strategy == "ivp":
            left_alpha = (left_var/(left_var + right_var))
            right_alpha = 1 - left_alpha

        elif self.strategy == "erc":
            left_alpha = (math.sqrt(left_var/right_var))/(1+math.sqrt(left_var/right_var))
            right_alpha = 1 - left_alpha

        return left_alpha, right_alpha

    def get_cluster_cov_matrix(self, leafs):
        """
        Returns the covariance matrix for the timeseries contained in ``leafs``.

        Parameters
        ----------
        leafs: list of int
            The indices of the leafs within a dendrogram

        Returns
        -------
        ndarray
        """
        var_cov = self.quasi_diag_cov[[self.tickers[idx] for idx in leafs]]
        var_cov = var_cov.reindex([self.tickers[idx] for idx in leafs])
        
        return var_cov.to_numpy()

    def get_weights(self, var_cov):
        """
        Returns the weight allocations of securities within a cluster. The weight allocation depends
        on the strategy being used. For all strategies, weights must be positive (long-only constraint) and 
        sum to one (full-investment constraint).

        if ``self.strategy``=="minvar"
        ------------------------------
        Returns weights such that the variance of the portfolio is minimized.

        if ``self.strategy``=="erc"
        ------------------------------
        Returns weights such that each security within the cluster contributes an equal amount of risk
        to the total risk of that cluster.

        if ``self.strategy``=="ivp"
        ------------------------------
        Each security within a cluster recieves an allocation inversely proportional to its volatility.

        Parameters
        ----------
        ndarray
            The quasi-diagonal variance-covariance matrix for a cluster

        Returns 
        -------
        ndarray
        """
        if (self.strategy =="minvar"):
            weights = self.minimum_variance_optimization(var_cov)

        elif (self.strategy=="ivp"):
            var_cov_diag = np.diag(np.diag(var_cov))
            w_numerator = np.diag(np.linalg.inv(var_cov_diag))
            w_denominator = np.trace(np.linalg.inv(var_cov_diag))
            weights = w_numerator/w_denominator

        elif (self.strategy=="erc"):
            weights = self.ERC_optimization(var_cov)

        return weights
  
    def get_all_leafs(self, cluster_num):
        """
        Returns all leafs (original observations) within a cluster. This cluster
        is identified by cluster_num.

        Parameters
        ----------
        cluster_num: int
            The integer associated with the cluster as outlined in the documentation of
            scipy.cluster.hierarchy.linkage
        
        Returns
        -------
        list of int
            A list of cluster numbers, where the numbers represent the original observations

        Note: if cluster_number < n, then the cluster is an original observation. 
        """
        n = len(self.link_matrix) + 1
        leaves = []

        if cluster_num < n:
            leaves.append(cluster_num)
            return leaves
        else:
            link_matrix_idx = cluster_num - n
            left_cluster = int(self.link_matrix[link_matrix_idx, 0])
            right_cluster = int(self.link_matrix[link_matrix_idx, 1])
            return self.get_all_leafs(left_cluster) + self.get_all_leafs(right_cluster) 

    def minimum_variance_optimization(self, var_cov):
        """
        Optimization for a minimum-variance portfolio.

        Parameters
        ----------
        var_cov: ndarray
            The variance-covariance matrix for a cluster

        Returns 
        -------
        float
        """
        def min_var_function(w):
            return 1/2*(w.T @ var_cov @ w)

        def jac_min_var_function(w):
            return var_cov @ w

        bounds = Bounds(0, 1)
        w0 = np.ones((var_cov.shape[0]))*1/var_cov.shape[0]
        eq_cons = {'type': 'eq', 
                   'fun': lambda w: np.array([np.sum(w) - 1]),
                   'jac': lambda w: np.array(np.ones_like(w))}
        ineq_cons = {'type': 'ineq', 
                   'fun': lambda w: np.array(list(w)),
                   'jac': lambda w: np.eye(var_cov.shape[0])} 
        opt = minimize(min_var_function, w0, method='SLSQP', 
                        jac=jac_min_var_function, 
                        constraints=[eq_cons, ineq_cons], 
                        options={'ftol':1e-9, 'disp': False},
                        bounds=bounds)
        return opt.x
        

    def ERC_optimization(self, var_cov):
        """
        Optimization for the ERC weight algorithm.

        Parameters
        ----------

        var_cov: ndarray
            The variance-covariance matrix for a cluster

        Returns 
        -------
        float
        """
        def inner_ERC_min_function(w):
            return self.ERC_min_function(w, var_cov)

        def inner_jac_ERC_min_function(w):
            return self.jac_ERC_min_function(w, var_cov)

        bounds = Bounds(0, 1)
        w0 = np.ones((var_cov.shape[0]))*1/var_cov.shape[0]
        eq_cons = {'type': 'eq', 
                   'fun': lambda w: np.array([np.sum(w) - 1]),
                   'jac': lambda w: np.array(np.ones_like(w))}
        ineq_cons = {'type': 'ineq', 
                   'fun': lambda w: np.array(list(w)),
                   'jac': lambda w: np.eye(var_cov.shape[0])} 
        opt = minimize(inner_ERC_min_function, w0, method='SLSQP', 
                        jac=inner_jac_ERC_min_function, 
                        constraints=[eq_cons, ineq_cons], 
                        options={'ftol':1e-9, 'disp': False},
                        bounds=bounds)
        return opt.x

    def ERC_min_function(self, w, var_cov):
        """
        The sum of squared pair-wise differences in marginal risk contributions for the ERC algorithm.
        This function is minimized in ERC.

        Parameters
        ----------
        w: ndarray
            The weight allocations

        var_cov: ndarray
            The variance-covariance matrix for a cluster

        Returns 
        -------
        float
        """
        sum = 0
        for i in range(len(w)):
            for j in range(len(w)):
                sum += (w[i]*(var_cov @ w)[i] - w[j]*(var_cov @ w)[j])**2
        return sum

    def jac_ERC_min_function(self, w, var_cov):
        """
        The jacobian of the sum of squared pair-wise differences in marginal risk contributions for the ERC algorithm.

        Parameters
        ----------
        w: ndarray
            The weight allocations

        var_cov: ndarray
            The variance-covariance matrix for a cluster

        Returns 
        -------
        ndarray
        """
        
        n = len(w)
        jac = np.zeros_like(w)
        
        for l in range(n):
            
            # NESTED SUM
            nested_sum = 0
            for k in range(n):
                if k == l:
                    continue
                else:
                    nested_sum += var_cov[l, k]*w[k]

            # FIRST SUM
            first_sum = 0
            for j in range(n):
                if j == l:
                    continue
                else:
                    first_sum += (w[l]*(var_cov @ w)[l] - w[j]*(var_cov @ w)[j]) * ((nested_sum + 2*var_cov[l,l]*w[l]) - var_cov[j,l]*w[j])
                    first_sum = first_sum*2
  
            # SECOND SUM
            second_sum = 0
            for i in range(n):
                if i == l:
                    continue
                else:
                    second_sum += (w[i]*(var_cov @ w)[i] - w[l]*(var_cov @ w)[l]) * (var_cov[i, l] * w[i] - (nested_sum + 2*var_cov[l,l]*w[l]))
                    second_sum = second_sum*2
        
            # THIRD SUM
            third_sum = 0
            for p in range(n):
                if p == l:
                    continue
                else:
                    for q in range(n):
                        if q == l:
                            continue
                        else:
                            third_sum += (w[p]*(var_cov @ w)[p] - w[q]*(var_cov @ w)[q]) * (var_cov[p, l]*w[p] - var_cov[q, l]*w[q])           
            third_sum = third_sum*2
            
            jac[l] = first_sum + second_sum + third_sum

        return jac



