from numpy import loadtxt, log
from pandas import read_csv
from scipy.stats import multivariate_normal
from distances import dv_over_rs, dm_over_rs, dh_over_rs

data = read_csv("bao_data/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
                header=None, index_col=None, sep=r"\s+", comment="#")
cov = loadtxt("bao_data/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt")

zs = data.iloc[:, 0].to_numpy()
mean = data.iloc[:, 1].to_numpy()
di_over_rss = [getattr(distances, i.lower()) for i in data.iloc[:, 2]]

gaussian = multivariate_normal(mean, cov)


def likelihood(theta, omegar=0):
    h0rd = theta[0]
    omegam = theta[1]
    theta = theta[2:]
    x = [
        di_over_rs(z, h0rd, omegam, omegar, theta)
        for z, di_over_rs in zip(zs, di_over_rss)
    ]
    likelihood = gaussian.pdf(x)
    if likelihood == 0:
        return -1e30
    return log(likelihood)
