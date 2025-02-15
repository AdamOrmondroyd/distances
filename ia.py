import sys
import numpy as np
import pandas as pd
from pypolychord.priors import UniformPrior
from common import run
from distances import dl, dl_no_h0, c
from pathlib import Path


# data loading stolen from Toby
path = Path('../clik_installs/desi/data/sn_data/PantheonPlus')
df = pd.read_table(path/'Pantheon+SH0ES.dat', sep=' ', engine='python')
cov = np.loadtxt(path/'Pantheon+SH0ES_STAT+SYS.cov', skiprows=1)
cov = cov.reshape([-1, int(np.sqrt(len(cov)))])

mask = df['zHD'] > 0.023
cepheid_mask = (df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)
ia_mask = (df['IS_CALIBRATOR'] == 0)[cepheid_mask]


def mb_zs_mcov(df, mask):
    mb = df['m_b_corr'].to_numpy()[mask]
    zhd = df['zHD'].to_numpy()[mask]
    zhel = df['zHEL'].to_numpy()[mask]
    mcov = cov[mask, :][:, mask]
    one = np.ones(len(mcov))[:, None]
    invcov = np.linalg.inv(mcov)
    invcov_tilde = invcov - invcov @ one @ one.T @ invcov / (one.T @ invcov @ one)
    lognormalisation = 0.5 * (np.log(2*np.pi)
                              - np.linalg.slogdet(2 * np.pi * mcov)[1]
                              - np.log((one.T @ invcov @ one).squeeze()))

    return mb, zhd, zhel, mcov, invcov, invcov_tilde, lognormalisation


mb, zhd, zhel, mcov, invcov, invcov_tilde, lognormalisation = mb_zs_mcov(df, mask)
mb_c, zhd_c, zhel_c, mcov_c, invcov_c, invcov_tilde_c, lognormalisation_c = mb_zs_mcov(df, cepheid_mask)


cephdist = df['CEPH_DIST'].to_numpy()[cepheid_mask]
delta_c = mb_c - cephdist
cephmask = 1 - ia_mask
delta_c_masked = delta_c * cephmask


h0min, h0max = 20, 100

# precompute some matrices
one = np.ones(len(mcov))[:, None]
onesigma_times_5_over_log10 = one.T @ invcov_tilde * 5 / log(10)
new_lognormalisation = lognormalisation + log(c/(1e-5 * (h0max - h0min)))
a = 1e-5 * h0min / c
b = 1e-5 * h0max / c


def logl_ia(omegam, omegar, theta=np.array([-1])):
    theta = np.array(theta)
    y = 5 * log10(h0_dl_over_c(zhd, zhel, omegam, omegar, theta)) - mb
    capital_y = float((onesigma_times_5_over_log10 @ y).squeeze())
    return (
        - float(y.T @ invcov_tilde @ y / 2)
        + log((b**(capital_y + 1) - a**(capital_y + 1)) / (capital_y + 1))
        + new_lognormalisation)


def logl_cepheid(h0, omegam, omegar, theta=np.array([-1])):
    theta = np.array(theta)
    mu = 5 * np.log10(dl(zhd_c, zhel_c, h0, omegam, omegar, theta)) + 25
    x = mb_c - mu
    x = (x * ia_mask + delta_c_masked)
    return lognormalisation_c + float(-x.T @ invcov_tilde_c @ x / 2)


omegar = 8.24e-5

if __name__ == "__main__":

    if len(sys.argv) > 2 and "cepheid" in sys.argv[2]:
        raise NotImplementedError("Still need to marginalise cepheid likelihood over H0")
        def likelihood(theta):
            h0, omegam, *theta = theta
            return logl_cepheid(h0, omegam, omegar, theta)

        ns = run(
            likelihood,
            sys.argv[1],
            [UniformPrior(0.01, 0.99)],
            "cepheid",
            [("Omegam", r"\Omega_\mathrm{m}")],
            True,
        )

    else:
        def likelihood(theta):
            omegam, *theta = theta
            return logl_ia(omegam, omegar, theta)

        ns = run(
            likelihood,
            sys.argv[1],
            [UniformPrior(0.01, 0.99)],
            "ia",
            [("Omegam", r"\Omega_\mathrm{m}")],
            True,
        )
