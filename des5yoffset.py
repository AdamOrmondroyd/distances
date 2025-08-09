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


# george fiddle
des_id = 10.0
george_mask = df['IDSURVEY'] != des_id


class GeorgeIaLogL(IaLogL):
    def __init__(self, george_mask, *args, **kwargs):
        self.george_mask = george_mask
        super().__init__(*args, **kwargs)

    def _y(self, delta_mb, *args, **kwargs):
        # offset is the george offset
        y = super()._y(*args, **kwargs)
        y[self.george_mask] -= delta_mb
        return y


logl_des5y = GeorgeIaLogL(george_mask, df, cov, mb_column='MU', z_cutoff=0.0)

omegar = 8.24e-5

if __name__ == "__main__":

    def likelihood(theta):
        delta_mb, omegam, *theta = theta
        return logl_des5y(delta_mb, omegam, omegar, theta)

    ns = run(
        likelihood,
        sys.argv[1],
        [
            UniformPrior(-0.1, 0.1),
            UniformPrior(0.01, 0.99),
         ],
        "des5yoffset",
        [
            ("delta_mb", r"\Delta m_\mathrm{B}"),
            ("Omegam", r"\Omega_\mathrm{m}"),
        ],
        read_resume=True,
    )
