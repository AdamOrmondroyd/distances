import sys
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from common import run
from distances import dl


# data loading stolen from Toby
df = pd.read_table('../clik_installs/desi/data/sn_data/PantheonPlus/Pantheon+SH0ES.dat', sep=' ', engine='python')
cov = np.reshape(np.loadtxt('../clik_installs/desi/data/sn_data/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov', skiprows=1), [1701, 1701])

mask = (df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)

mbcorr = df['m_b_corr'].to_numpy()[mask]

zhd = df['zHD'].to_numpy()[mask]
zhel = df['zHEL'].to_numpy()[mask]

cephdist = df['CEPH_DIST'].to_numpy()[mask]

mcov = cov[mask, :][:, mask]

gaussian = multivariate_normal(np.zeros(len(mcov)), mcov)


one = np.ones(len(mcov))[:, None]
invcov = np.linalg.inv(mcov)
invcov_tilde = invcov - invcov @ one @ one.T @ invcov / (one.T @ invcov @ one)
lognormalisation = 0.5 * (np.log(2*np.pi)
                          - np.linalg.slogdet(2 * np.pi * mcov)[1]
                          - float(np.log(one.T @ invcov @ one)))


def logl_ia(h0, omegam, omegar, theta=np.array([-1])):
    theta = np.array(theta)
    mu = 5 * np.log10(dl(zhd, zhel, h0, omegam, omegar, theta)) + 25
    x = mbcorr - mu

    return lognormalisation + float(-x.T @ invcov_tilde @ x / 2)


omegar = 8.24e-5

if __name__ == "__main__":

    def likelihood(theta):
        h0, omegam, *theta = theta
        return logl_ia(h0, omegam, omegar, theta)

    ns = run(
        likelihood,
        sys.argv[1],
        [20, 0.01],
        [100, 0.99],
        "ia",
        [("H0", r"H_0"), ("Omegam", r"\Omega_\mathrm{m}")],
        True,
    )
