# PyHRP

## Summary

PyHRP is a library for hierarchical risk-based portfolios which allows users to create full-investment, long-only portfolios using techniques and strategies as outlined in "Hierarchical risk parity: Accounting for tail dependencies in multi-asset multi-factor allocations" by Harald Lohre et al. 

## The Algorithm

The algorithm which this library is based upon is taken directly from Lohre et al.

1. Perform Hierarchical clustering and generate a dendrogram
2. Assign all assets a unit weight &omega;<sub>i</sub> = 1 &forall;i = 1,...,N
3. For each dendrogram node (beginning from the top): 
    1. Determine the members of the clusters C<sub>1</sub> and C<sub>2</sub> belonging to the two sub-branches of the according dendrogram node
    2. Calculate the within-cluster allocations &omega;'<sub>1</sub> and &omega;'<sub>2</sub> for C<sub>1</sub> and C<sub>2</sub> according to the chosen methodology (more on this below)
    3. Based on the within-cluster allocations &omega;'<sub>1</sub> and &omega;'<sub>2</sub> calculate the across-cluster allocation &alpha; (splitting factor) for C<sub>1</sub> and (1-&alpha;) for C<sub>2</sub>
    4. For each asset in C<sub>1</sub> re-scale allocation &omega; by factor &alpha;
    5. For each asset in C<sub>2</sub> re-scale allocation &omega; by factor (1-&alpha;)
4. End   
    
## Hierarchical Clustering

Hierarchical clustering is first performed in order to group together timeseries in a dendrogram which are (a) highly correlated or (b) share similar correlation structures across the entire portfolio or (c) are highly likely to experience downturns simultaneously (more on this below). The resulting dendrogram will result in a binary tree looking something like this:

![](images/dendrogram.png)

### The Dissimilarity Matrix

The dissimilarity matrix is an integral component to generating dendrograms like the one above. The dissimilarity matrix explains how "dissimilar" each pair of timeseries within our universe are. PyHRP provides implementations for three different types of dissimilarity matrices to be chosen by the user.

#### CorrDissimilarity
A dissimilarity matrix based on 'CorrDissimilarity' dissimilarities has its (i,j)<sup>th</sup> element equal to ((1-p<sub>i,j</sub>)/2)<sup>1/2</sup>, where p<sub>i,j</sub> is the correlation between timeseries i and j. Hence, positively correlated timeseries are deemed relatively 'similar', while negatively correlated timeseries are deemed relatively 'dissimiliar'. 

#### PortfolioDissimilarity
A dissimilarity matrix based on 'Portfolio' dissimilarities has its (i,j)<sup>th</sup> element equal to l2-norm of the difference between vectors of 'CorrDissimilarities' belonging to the i<sup>th</sup> timeseries and the j<sup>th</sup> timeseries. This dissimilarity metric takes into account not only how two timeseries are correlated, but how they correlate to the rest of the portfolio. 

#### LTDCDissimilarity
A dissimilarity matrix based on lower-tail-dependency coefficients (LTDCs) between pairs of timeseries. The LTDC of timeseries i and j is defined by the limit of
&lambda;(c) = P(X<sub>i</sub> < c | X<sub>j</sub> < c) as c &rarr; &infin;. In order to estimate the LTDC, PyHRP uses psuedo-maximum-likelihood estimation on the bivariate Clayton copula. This dissimilarity metric is often seen as more robust than the previous two as LTDCs don't rely on correlations. Correlations between two types of stocks can be quite volatile, and tend to increase dramatically during market downturns. In order to use this metric, the user must be equipped with timeseries data that is roughly iid. Since financial timeseries data is almost never iid, some level of preprocessing is required when using LTDCs. Giovanni De Luca and Paola Zuccolotto use the standardized residuals of an AR(1)-GARCH(1,1) model in order to accomplish this in their paper referenced below.

### Linkage Method
Constructing a dendrogram requires the user to specify a linkage method. In order to make an educated decision as to what linkage method is best, I would skim through the resources below. Pitfalls can exist when using certain linkage methods with certain dissimilarity matrices. For example, documentation for the SciPy linkage method used in PyHRP states that "methods ‘centroid’, ‘median’, and ‘ward’ are correctly defined only if a Euclidean pairwise metric is used". In PyHRP, PortfolioDissimilarity is the only Euclidean pairwise metric. The user can implement their own dissimilarity metric if they wish to do so.

## Calculating within-cluster allocations
PyHRP provides the user with three different risk-based strategies to compute within-cluster allocations.

### The Minimum Variance Portfolio (MVP)
When computing allocations under this strategy, we use numerical optimization to find **w**<sup>*</sup> = argmin<sub>**w** </sub>**w**<sup>T</sup>V**w** under the constraint that all weights sum to one (full-investment) and are GEQ 0 (long-only), where V is the variance-covariance matrix of the cluster.

### The Inverse Volatility Portfolio (IVP)
If we were to ignore the correlations within the cluster, we would end up with an inverse volatility portfolio (IVP). Each allocation is the inversely proportional to the volatility of the underlying timeseries. This strategy may be ideal if we want to ignore correlations for the same reasons outlined in the discussion about LTDCs. 

### The Equal Risk Contribution Portfolio (ERC)
The ERC portfolio is simply a portfolio where all timeseries in a cluster contribute the same amount of risk to the overall portfolio. This is different from IVP since we take correlations into account (setting correlations to 0 once again leads us to the IVP portfolio). In order to compute allocations under ERC, we use numerical optimization to find 
**w**<sup>*</sup> = argmin<sub>**w**</sub> &Sigma;<sub>i,j</sub> (w<sub>i</sub> MRC<sub>i</sub> - w<sub>j</sub> MRC<sub>j</sub>)<sup>2</sup>, where MRC<sub>l</sub> is the marginal risk contribution of timeseries l, or the partial derivative of the total portfolio risk with respect to w<sub>l</sub>.

## Calculating across-cluster allocations
PyHRP computes splitting factors according to the methodology that was used for the within-cluster allocation. If MVP was used to compute within-cluster allocation, then the splitting factor is computed such that the total variance between two clusters is minimized; the covariance between the two clusters is taken into account. If IVP was used, then the splitting factor is computed such that each cluster is weighted inversely proportional to its total variance; this is equivalent to minimizing the total variance between the two clusters while ignoring the covariance between the two clusters. If ERC was used, the the splitting factor is computed such that both clusters have the same total variance after being weighted by the splitting factor.

## Resources
1. Harald Lohre, Carsten Rother, and Kilian Axel Schafer, "Hierarchical Risk Parity: Accounting for tail dependencies in multi-asset multi-factor allocations",  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3513399
2. Daniel Mullner, “Modern hierarchical, agglomerative clustering algorithms”, https://arxiv.org/abs/1109.2378v1
3. Documentation for the hierarchical clustering algorithm used in PyHRP:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
4. Giovanni De Luca, Paola Zuccolotto, "A tail dependence-based dissimilarity measure for financial time series clustering", https://link.springer.com/article/10.1007/s11634-011-0098-3
5. Marcos Lopez De Prado, "Building Diversified Portfolios that Outperform Out of Sample", https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
6. Sebastien Maillard, Thierry Roncalli, and Jerome Teiletche, "On the Properties of Equally Weighted Risk Contribution Portfolios", https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1271972
7. Erick Forseth and Ed Tricker, "Equal Risk Contribution Portfolios", https://www.grahamcapital.com/Equal%20Risk%20Contribution%20April%202019.pdf
