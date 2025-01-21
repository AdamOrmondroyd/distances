import sys
import numpy as np
import pandas as pd
from pypolychord.priors import UniformPrior
from common import run
from distances import dl
from pathlib import Path


# data loading stolen from Toby
path = Path('../clik_installs/desi/data/sn_data/DESY5')
df = pd.read_table(path/'DES-SN5YR_HD.csv', sep=',', engine='python').sort_values("zHD")
print(df)
cov = np.loadtxt(path/'covsys_000.txt', skiprows=1)
cov = cov.reshape([-1, int(np.sqrt(len(cov)))])

# mask = df['zHD'] > 0.023
mask = np.full(len(df), True)


def mb_zs_mcov(df, mask):
    mb = df['MU'].to_numpy()[mask]
    zhd = df['zHD'].to_numpy()[mask]
    zhel = df['zHEL'].to_numpy()[mask]
    delta = df['MUERR_FINAL'].to_numpy()[mask]
    mcov = cov[mask, :][:, mask]
    np.fill_diagonal(mcov, delta**2 + mcov.diagonal())
    one = np.ones(len(mcov))[:, None]
    invcov = np.linalg.inv(mcov)
    invcov_tilde = invcov - invcov @ one @ one.T @ invcov / (one.T @ invcov @ one)
    lognormalisation = 0.5 * (np.log(2*np.pi)
                              - np.linalg.slogdet(2 * np.pi * mcov)[1]
                              - np.log((one.T @ invcov @ one).squeeze()))

    return mb, zhd, zhel, mcov, invcov, invcov_tilde, lognormalisation


mb, zhd, zhel, mcov, invcov, invcov_tilde, lognormalisation = mb_zs_mcov(df, mask)


def logl_des5y(h0, omegam, omegar, theta=np.array([-1])):
    theta = np.array(theta)
    mu = 5 * np.log10(dl(zhd, zhel, h0, omegam, omegar, theta)) + 25
    x = mb - mu

    return lognormalisation + float(-x.T @ invcov_tilde @ x / 2)


omegar = 8.24e-5

if __name__ == "__main__":

    def likelihood(theta):
        h0, omegam, *theta = theta
        return logl_des5y(h0, omegam, omegar, theta)

    ns = run(
        likelihood,
        sys.argv[1],
        [UniformPrior(20, 100), UniformPrior(0.01, 0.99)],
        "des5yr",
        [("H0", r"H_0"), ("Omegam", r"\Omega_\mathrm{m}")],
        True,
    )
