import sys
import numpy as np
import pandas as pd
from pypolychord.priors import UniformPrior
from common import run
from distances import dl


# data loading stolen from Toby
df = pd.read_table('../clik_installs/desi/data/sn_data/PantheonPlus/Pantheon+SH0ES.dat', sep=' ', engine='python')
cov = np.reshape(np.loadtxt('../clik_installs/desi/data/sn_data/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov', skiprows=1), [1701, 1701])

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


def logl_ia(h0, omegam, omegar, theta=np.array([-1])):
    theta = np.array(theta)
    mu = 5 * np.log10(dl(zhd, zhel, h0, omegam, omegar, theta)) + 25
    x = mb - mu

    return lognormalisation + float(-x.T @ invcov_tilde @ x / 2)


def logl_cepheid(h0, omegam, omegar, theta=np.array([-1])):
    theta = np.array(theta)
    mu = 5 * np.log10(dl(zhd_c, zhel_c, h0, omegam, omegar, theta)) + 25
    x = mb_c - mu
    x = (x * ia_mask + delta_c_masked)
    return lognormalisation_c + float(-x.T @ invcov_tilde_c @ x / 2)


omegar = 8.24e-5

if __name__ == "__main__":

    if len(sys.argv) > 2 and "cepheid" in sys.argv[2]:
        def likelihood(theta):
            h0, omegam, *theta = theta
            return logl_cepheid(h0, omegam, omegar, theta)

        ns = run(
            likelihood,
            sys.argv[1],
            [UniformPrior(20, 100), UniformPrior(0.01, 0.99)],
            "cepheid",
            [("H0", r"H_0"), ("Omegam", r"\Omega_\mathrm{m}")],
            True,
        )

    else:
        def likelihood(theta):
            h0, omegam, *theta = theta
            return logl_ia(h0, omegam, omegar, theta)

        ns = run(
            likelihood,
            sys.argv[1],
            [UniformPrior(20, 100), UniformPrior(0.01, 0.99)],
            "ia",
            [("H0", r"H_0"), ("Omegam", r"\Omega_\mathrm{m}")],
            True,
        )
