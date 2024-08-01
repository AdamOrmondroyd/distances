from numpy import loadtxt, log
from pandas import read_csv
from scipy.stats import multivariate_normal
import distances

data = read_csv("bao_data/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
                header=None, index_col=None, sep=r"\s+", comment="#")
cov = loadtxt("bao_data/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt")

zs = data.iloc[:, 0].to_numpy()
mean = data.iloc[:, 1].to_numpy()
di_over_rss = [getattr(distances, i.lower()) for i in data.iloc[:, 2]]

gaussian = multivariate_normal(mean, cov)


def desi_likelihood(h0rd, omegam, omegar, theta):
    x = [
        di_over_rs(z, h0rd, omegam, omegar, theta)
        for z, di_over_rs in zip(zs, di_over_rss)
    ]
    return gaussian.logpdf(x)
