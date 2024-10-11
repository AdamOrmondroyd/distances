import sys
import numpy as np
from scipy.stats import multivariate_normal
from pandas import read_csv
from pypolychord.priors import UniformPrior
from common import run
import distances

data = read_csv("../clik_installs/desi/data/bao_data/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
                header=None, index_col=None, sep=r"\s+", comment="#")
cov = np.loadtxt("../clik_installs/desi/data/bao_data/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt")

zs = data.iloc[:, 0].to_numpy()
mean = data.iloc[:, 1].to_numpy()
di_over_rss = [getattr(distances, i.lower()) for i in data.iloc[:, 2]]

gaussian = multivariate_normal(mean, cov)


def logl_desi(h0rd, omegam, omegar, theta):
    theta = np.array(theta)
    x = [
        di_over_rs(z, h0rd, omegam, omegar, theta)
        for z, di_over_rs in zip(zs, di_over_rss)
    ]
    return gaussian.logpdf(x)


omegar = 8.24e-5

if __name__ == "__main__":

    def likelihood(theta):
        h0rd, omegam, *theta = theta
        return logl_desi(h0rd, omegam, omegar, theta)

    ns = run(
        likelihood,
        sys.argv[1],
        [UniformPrior(3650, 18250), UniformPrior(0.01, 0.99)],
        "desi",
        [(r"H0rd", r"H_0r_\mathrm{d}"), (r"Omegam", r"\Omega_\mathrm{m}")],
        True,
    )
