import sys
import numpy as np
import pandas as pd
from pypolychord.priors import UniformPrior
from common import run
from pathlib import Path
from ia import IaLogL


# data loading stolen from Toby
path = Path('../clik_installs/desi/data/sn_data/DESY5')
df = pd.read_table(path/'DES-SN5YR_HD.csv', sep=',', engine='python')
# DES 5Y data is not sorted by redshift
idx = np.argsort(df['zHD'])
print(f"{idx=}")
print(df)
cov = np.loadtxt(path / 'covsys_000.txt', skiprows=1)
cov = cov.reshape([-1, int(np.sqrt(len(cov)))])
delta = df['MUERR_FINAL'].to_numpy()
np.fill_diagonal(cov, delta**2 + cov.diagonal())
cov = cov[idx, :][:, idx]
df = df.iloc[idx]

logl_des5y = IaLogL(df, cov, mb_column='MU', z_cutoff=0.0)

omegar = 8.24e-5

# print(f"highest redshift is {np.max(df['zHD']):.2f}.",
      # f"Cutting off at a={1/(1+np.max(df['zHD'])):.3f}",
      # flush=True)

if __name__ == "__main__":

    def likelihood(theta):
        omegam, *theta = theta
        return logl_des5y(omegam, omegar, theta)

    ns = run(
        likelihood,
        sys.argv[1],
        [UniformPrior(0.01, 0.99)],
        # "des5y_a_cutoff",
        "des5y",
        [("Omegam", r"\Omega_\mathrm{m}")],
        read_resume=True,
        wide_prior=sys.argv[1] != "cpl",
        # low_a=1/(1+np.max(df['zHD'])),
    )
