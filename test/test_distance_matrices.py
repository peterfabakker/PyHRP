from ..pyhrp.tools.distancematrices import CorrDistance, PortfolioDistance, LTDCDistance
from math import fabs, sqrt, log
import pandas as pd
import numpy as np

def test_CorrDistance_get_distance_matrix():
    d = {'A': [0.5, 1, 0], 'B': [0, -1, 0.5], 'C':[-0.5, 1, 0]}
    df = pd.DataFrame(data=d)
    corr = df.corr()

    zoc = CorrDistance(corr)
    zoc_matrix = zoc.get_distance_matrix()

    assert fabs(zoc_matrix.iloc[0,0] - 0) < 1e-5
    assert fabs(zoc_matrix.iloc[1,0] - 0.995485057) < 1e-5
    assert fabs(zoc_matrix.iloc[2,0] - 0.415539408) < 1e-5
    assert fabs(zoc_matrix.iloc[2,1] - 0.944911106) < 1e-5

def test_PortfolioDistance_get_distance_matrix():
    d = {'A': [0.5, 1, 0], 'B': [0, -1, 0.5], 'C':[-0.5, 1, 0]}
    df = pd.DataFrame(data=d)
    corr = df.corr()

    portd = PortfolioDistance(corr)
    portd_matrix = portd.get_distance_matrix()

    assert fabs(portd_matrix.iloc[0,0] - 0) < 1e-5
    assert fabs(portd_matrix.iloc[0,1] - sqrt(((0 - 0.995485057)**2  + (0.995485057 - 0)**2 + (0.415539408 - 0.944911106)**2))) < 1e-5
    
def test_ZOCDistance_get_compressed_distance_matrix():
    d = {'A': [0.5, 1, 0], 'B': [0, -1, 0.5], 'C':[-0.5, 1, 0]}
    df = pd.DataFrame(data=d)
    corr = df.corr()

    zoc = CorrDistance(corr)
    zoc_matrix = zoc.get_distance_matrix()
    zoc_matrix_compressed = zoc.get_compressed_distance_matrix()
    m = zoc_matrix.shape[0]

    for i in range(0, m):
        for j in range(i+1, m):
            assert fabs(zoc_matrix.iloc[i,j] - zoc_matrix_compressed[int(m*i + j - (((i+2)*(i+1))/2))]) < 1e-5
    
def test_PORTDistance_get_compressed_distance_matrix():
    d = {'A': [0.5, 1, 0], 'B': [0, -1, 0.5], 'C':[-0.5, 1, 0]}
    df = pd.DataFrame(data=d)
    corr = df.corr()

    portd = PortfolioDistance(corr)
    portd_matrix = portd.get_distance_matrix()
    portd_matrix_compressed = portd.get_compressed_distance_matrix()
    m = portd_matrix.shape[0]

    for i in range(0, m):
        for j in range(i+1, m):
            assert fabs(portd_matrix.iloc[i,j] - portd_matrix_compressed[int(m*i + j - (((i+2)*(i+1))/2))]) < 1e-5
    
def test_get_empirical_CDF():
    timeseries = [-1, 0, 1]
    ltdc_dist = LTDCDistance(timeseries, copula='clayton')
    
    assert fabs(ltdc_dist.get_empirical_CDF(timeseries, 0.5) - 0.5) < 0.001

def test_clayton_log_likelihood():
    timeseries = pd.DataFrame([[-1, 1],
                                [0, 2]])
    ltdc_dist = LTDCDistance(timeseries, 'clayton')
    L = 2*log(2) - 3*log(10) + 4*log(9/2)
    
    assert fabs(ltdc_dist.clayton_log_likelihood(theta=1, i=0, j=1) - L) < 0.001

def test_clayton_d_log_likelihood():
    timeseries = pd.DataFrame([[-1, 1],
                                [0, 2]])
    ltdc_dist = LTDCDistance(timeseries, 'clayton')
    g_0 = (2*(3) - 1)
    g_1 = (2*(3/2) - 1)
    dg_0 = 2*(3)*log(3) 
    dg_1 = 2*(3/2)*log(3/2) 
    dL = 1 - 3*(dg_0/g_0 + dg_1/g_1) + log(g_0) + log(g_1) - 2*log(1/3) - 2*log(2/3)

    assert fabs(ltdc_dist.clayton_d_log_likelihood(theta=1, i=0, j=1) - dL) < 0.001