from .preprocessing import amalgamate_prices, parse_yahoo_prices
from .pyhrp.hierarchicalriskportfolios import HierarchicalRiskPortfolio
from .test import test_distance_matrices
from .test import test_quasidiag
from .test import test_weightallocation
import pandas as pd

PATH = ''
PATH_LTDC = ''

def main():
    """
    Main function. Choose from functions below.
    """
    sandbox()

def ltdc_sandbox():
    std_res_returns = pd.read_csv(PATH_LTDC, index_col=0).iloc[:,1:3]
    hrp = hrp = HierarchicalRiskPortfolio('ltdc', 'average', 'ivp', std_res_returns)
    print(hrp.get_allocations())

def sandbox():
    """
    Sandbox. Just used for quick testing of packages.
    """
    prices = amalgamate_prices(parse_yahoo_prices(PATH))
    returns = prices/prices.shift(1) - 1
    returns = returns.iloc[1:, :]
    hrp = HierarchicalRiskPortfolio('portfolio', 'ward', 'ivp', returns)
    print(hrp.get_allocations())
   
def test_distance_package():
    """
    Run tests in test.test_distance_matrices.py.
    """
    test_distance_matrices.test_CorrDistance_get_distance_matrix()
    test_distance_matrices.test_PortfolioDistance_get_distance_matrix()
    test_distance_matrices.test_ZOCDistance_get_compressed_distance_matrix()
    test_distance_matrices.test_PORTDistance_get_compressed_distance_matrix()
    test_distance_matrices.test_get_empirical_CDF()
    test_distance_matrices.test_clayton_d_log_likelihood()
    test_distance_matrices.test_clayton_log_likelihood()
    
def test_quasidiag_package():
    """
    Run tests in test.test_quasidiag.py.
    """
    test_quasidiag.test_4by4_case()

def test_weightallocation_package():
    """
    Run tests in test.test_vanillaweightgen.py.
    """
    test_weightallocation.test_get_all_leafs()
    test_weightallocation.test_ERC_min_function()
    test_weightallocation.test_jac_ERC_min_function()
    test_weightallocation.test_ERC_optimization2()

if __name__ == '__main__':
    main()
   





