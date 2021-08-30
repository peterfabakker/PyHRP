"""
The module for constructing risk-based portfolios using hierarchical clustering. 
"""
from .tools.distancematrices import CorrDistance, PortfolioDistance, LTDCDistance
from .tools.quasidiag import quasidiagonalize_cov
from .tools.weightallocation import ClusteringAllocation
from scipy.cluster.hierarchy import linkage

class HierarchicalRiskPortfolio:
    """
    A class for constructing a hierarchical risk-based portfolio. Different risk-based portfolios can
    be constructed with the below parameters.

    Dissimilarity Metrics
    -------------------
    Pairs of timeseries can be ranked in dissimilarity by using 
    (a)  a dissimilarity metric based on the correlation between the pair
    (b)  a dissimiliarity metric based on how the two timeseries correlate with the entire portfolio
    (c)  a similarity metric based on the lower-tail dependecy coefficients of the two timeseries

    Linkage Methods
    --------------------
    A variety of linkage methods are supported by scipy.cluster.hierarchy.linkage. Note that the 'centroid', 'median', and 'ward' linkage methods
    are only well-defined when using the 'portfolio_dissimilarity' dissimilarity metric.

    Within and Across Cluster Allocation
    ------------------------------------
    This library as of now supports three allocation methods: minimum variance, inverse volatility, and equal risk contribution.
    The minimum variance allocation method takes into account correlations between different pairs of timeseries; ignoring these
    correlations leads to the inverse volatility portfolio. 
    """

    def __init__(self, dissimilarity, linkage, allocation_method, returns):        
        """
        Parameters
        ----------
        dissimilarity: str
            Can be set to one of 'corr', 'portfolio', or 'ltdc' for dissimilarity metrics
            (a), (b), and (c) respectively
        linkage: str
            See scipy.cluster.hierarchy.linkage for appropriate linkage methods and how they're defined mathematically
        allocation_method: str
            Can be set to one of 'minvar', 'ivp', or 'erc' for the minimum variance, inverse volatility, and equal risk
            contribution allocation methods respectively
        returns: pd.DataFrame
            A dataframe of timeseries of returns. If using the 'ltdc' dissimilarity measure, must be 
            preprocessed to remove autocorrelation and heteroscedasticity.
        """
        self.dissimilarity = dissimilarity
        self.linkage = linkage
        self.allocation_method = allocation_method
        self.returns = returns
        self.corr = returns.corr()

    def get_allocations(self):
        """
        Returns
        -------
        dict
        """
        if self.dissimilarity == 'corr':
            corr_distance = CorrDistance(self.corr)
            dissim = corr_distance.get_compressed_distance_matrix()
        elif self.dissimilarity == 'portfolio':
            portfolio_distance = PortfolioDistance(self.corr)
            dissim = portfolio_distance.get_compressed_distance_matrix()
        elif self.dissimilarity == 'ltdc':
            ltdc_distance = LTDCDistance(self.returns, 'clayton')
            dissim = ltdc_distance.get_compressed_distance_matrix()

        links = linkage(dissim, self.linkage)
        serialized_cov_matrix = quasidiagonalize_cov(self.returns, links)
        clusAl = ClusteringAllocation(serialized_cov_matrix, self.corr.columns.values, links, self.allocation_method, self.returns)

        return clusAl.get_portfolio_weights()
        


    

