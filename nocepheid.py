import sys
import numpy as np
from numpy import array, log10
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
import pypolychord
from distances import dl
from anesthetic import make_2d_axes
from mpi4py import MPI
from clustering import xmeans
from flexknot import Prior

comm = MPI.COMM_WORLD


# data loading stolen from Toby
df = pd.read_table('../clik_installs/desi/data/sn_data/PantheonPlus/Pantheon+SH0ES.dat', sep=' ', engine='python')
cov = np.reshape(np.loadtxt('../clik_installs/desi/data/sn_data/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov', skiprows=1), [1701, 1701])

mask = (df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)
iamask = (df['IS_CALIBRATOR'] == 0)[mask]

mbcorr = df['m_b_corr'].to_numpy()[mask]

z = df['zHD'].to_numpy()[mask]
zhd = df['zHD'].to_numpy()[mask]
zhel = df['zHEL'].to_numpy()[mask]

cephdist = df['CEPH_DIST'].to_numpy()[mask]

mcov = cov[mask, :][:, mask]

gaussian = multivariate_normal(np.zeros(len(mcov)), mcov)

mu2 = mbcorr - cephdist
cephmask = 1 - iamask
mu2masked = mu2 * cephmask


def ia_likelihood(Mb, h0, omegam, omegar, theta):
    theta = array(theta)

    # dl is calculated in Mpc = [c/H0]
    # mu = 5 log(dl/10pc), so 25 comes from converting from Mpc to pc
    mu = 5 * log10(dl(z, h0, omegam, omegar, theta)) + 25
    mu1 = mbcorr - mu

    mu = mu1
    return gaussian.logpdf(Mb - mu)


def corrected_ia_likelihood(Mb, h0, omegam, omegar, theta):
    theta = array(theta)

    # dl is calculated in Mpc = [c/H0]
    # mu = 5 log(dl/10pc), so 25 comes from converting from Mpc to pc
    mu = 5 * log10(dl(zhd, zhel, h0, omegam, omegar, theta)) + 25
    mu1 = mbcorr - mu

    mu = mu1
    return gaussian.logpdf(Mb - mu)


one = np.ones(len(mcov))[:, None]
invcov = np.linalg.inv(mcov)
invcov_tilde = invcov - invcov @ one @ one.T @ invcov / (one.T @ invcov @ one)
print(f"{mcov=}")
print(f"{invcov_tilde.shape=}")
print(f"{np.linalg.det(mcov)=}")
print(f"{scipy.linalg.det(mcov)=}")
print(f"{np.linalg.slogdet(mcov)=}")
print(f"{one.T @ invcov @ one=}")
lognormalisation = float(-0.5 * np.log(one.T @ invcov @ one))
print(f"{lognormalisation=}")
lognormalisation = 0.5 * (np.log(2*np.pi)
                          - np.linalg.slogdet(2 * np.pi * mcov)[1]
                          - float(np.log(one.T @ invcov @ one)))
print(f"{lognormalisation=}")


def marginalised(h0, omegam, omegar, theta):
    theta = np.array(theta)
    mu = 5 * log10(dl(z, h0, omegam, omegar, theta)) + 25
    mu1 = mbcorr - mu

    mu = (mu1 * iamask + mu2masked).to_numpy()[:, None]
    return lognormalisation + float(-mu.T @ invcov_tilde @ mu / 2)


lower = array([-20, 20, 0.01])
upper = array([-18, 100, 0.99])
# lower = array([20, 0.01])
# upper = array([100, 0.99])
prior_range = upper - lower

flexknotprior = Prior(0, 1, -3, -0.01)


def prior(x):
    return np.concatenate([
        lower + x[:len(lower)] * prior_range,
        flexknotprior(x[len(lower):])
    ])


omegar = 8.24e-5


def likelihood(theta):
    Mb, h0, omegam, *theta = theta
    return ia_likelihood(Mb, h0, omegam, omegar, theta)
    # return marginalised(h0, omegam, omegar, theta)


if __name__ == "__main__":
    try:
        n = int(sys.argv[1])
    except ValueError:
        n = 'lcdm'

        def prior(x):
            return lower + x * prior_range

        def likelihood(theta):
            Mb, h0, omegam = theta
            return ia_likelihood(Mb, h0, omegam, omegar, np.array([-1]))

    paramnames = [
        ("Mb", r"M_\mathrm{b}"),
        ("H0", r"H_0"),
        ("Omegam", r"\Omega_\mathrm{m}"),
    ]

    if n != "lcdm":
        if n >= 2:
            paramnames += [("wn", "w_n")]

        for i in range(n-2, 0, -1):
            paramnames += [
                (f"a{i}", f"a_{i}"),
                (f"w{i}", f"w_{i}"),
            ]
        if n >= 1:
            paramnames += [("w0", "w_0")]

    ndims = len(paramnames)
    file_root = f"nocepheid_{n}"

    for i in range(1):
        print(i)
        print(prior(np.random.rand(ndims)))
        print(likelihood(prior(np.random.rand(ndims))))

    ns = pypolychord.run(likelihood, ndims, prior=prior,
                         read_resume=False,
                         paramnames=paramnames,
                         file_root=file_root,
                         nlive=1000,
                         nprior=10_000,
                         cluster=xmeans,
                         )

    params = [p[0] for p in paramnames]
    if comm.rank == 0:
        fig, ax = make_2d_axes(params[:3])
        ns.plot_2d(ax)
        fig.savefig(f"plots/{file_root}.pdf", bbox_inches='tight')
