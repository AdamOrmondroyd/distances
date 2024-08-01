import numpy as np
from numpy import array, log10
import pandas as pd
from scipy.integrate import quad, cumulative_trapezoid
from scipy.stats import multivariate_normal
import pypolychord
from distances import c, h
from matplotlib import pyplot as plt
from anesthetic import make_2d_axes
from mpi4py import MPI

comm = MPI.COMM_WORLD


# data loading stolen from Toby
df = pd.read_table('pantheon1.txt', sep=' ', engine='python')
cov = np.reshape(np.loadtxt('Pantheon+SH0ES_STAT+SYS.cov.txt'), [1701, 1701])

mask = (df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)
cephmask = (df['IS_CALIBRATOR'] == 0)[mask]

mbcorr = df['m_b_corr'].to_numpy()[mask]

z = df['zHD'].to_numpy()[mask]

cephdist = df['CEPH_DIST'].to_numpy()[mask]

mcov = cov[mask, :][:, mask]

N = len(z)

gaussian = multivariate_normal(np.zeros(len(mcov)), mcov)

mu2 = mbcorr - cephdist
notcephmask = 1 - cephmask
mu2masked = mu2 * notcephmask


def dl(z, h0, omegam, omegar, theta):
    q0 = quad(lambda z: 1/h(z, omegam, omegar, theta), 0, z[0])[0]
    h_inverse = [1 / h(zi, omegam, omegar, theta) for zi in z]
    q = cumulative_trapezoid(h_inverse, z, initial=0) + q0
    return (1+z) * c / h0 * q


def ia_likelihood(Mb, h0, omegam, omegar, theta):

    mu = 5 * log10(dl(z, h0, omegam, omegar, theta)) + 25
    mu1 = mbcorr - mu

    mu = (mu1 * cephmask + mu2masked)
    return gaussian.logpdf(Mb - mu)


lower = array([-20, 50, 0.01])
upper = array([-18, 100, 0.99])
prior_range = upper - lower

flexknotprior = Prior(0, 1, -3, -0.01)


def prior(x):
    return np.concatenate([
        lower + x[:3] * prior_range,
        flexknotprior(x[3:])
    ])


def likelihood(theta):
    Mb = theta[0]
    h0 = theta[1]
    omegam = theta[2]
    omegar = 8.24e-5
    theta = np.array([-1])
    return ia_likelihood(Mb, h0, omegam, omegar, theta)

for i in range(1000):
    print(i)
    likelihood(prior(np.random.rand()))


paramnames = [
    ("Mb", r"M_\mathrm{b}"),
    ("H0", r"H_0"),
    ("Omegam", r"\Omega_\mathrm{m}"),
]

ns = pypolychord.run(likelihood, ndims, prior=prior,
                     read_resume=False,
                     paramnames=paramnames,
                     file_root="ia",
                     )

params = [p[0] for p in paramnames]
if comm.rank == 0:
    fig, ax = make_2d_axes(params)
    ns.plot_2d(ax)
    fig.savefig("plots/ia.pdf", bbox_inches='tight')
