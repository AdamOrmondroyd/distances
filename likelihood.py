import numpy as np
from pandas import read_csv
from scipy.stats import multivariate_normal
from distances import dv_over_rs, dm_over_rs, dh_over_rs

data = read_csv("bao_data/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
                header=None, index_col=None, sep=r"\s+", comment="#")
cov = np.loadtxt("bao_data/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt")

zs = data.iloc[:, 0].to_numpy()
mean = data.iloc[:, 1].to_numpy()
di_over_rss = [locals()[i.lower()] for i in data.iloc[:, 2]]

gaussian = multivariate_normal(mean, cov)


def likelihood(theta):
    x = np.array([
        di_over_rs(z, theta) for z, di_over_rs in zip(zs, di_over_rss)
    ])
    print(f'{x=}')
    return gaussian.pdf(x)
