import sys
import numpy as np
import pandas as pd
from pypolychord.priors import UniformPrior
from common import run
from pathlib import Path
from ia import IaLogL


# data loading stolen from Toby
path = Path('../clik_installs/desi/data/sn_data/PantheonPlus')
df = pd.read_table(path/'Pantheon+SH0ES.dat', sep=' ', engine='python')
cov = np.loadtxt(path/'Pantheon+SH0ES_STAT+SYS.cov', skiprows=1)
cov = cov.reshape([-1, int(np.sqrt(len(cov)))])

logl_pantheon = IaLogL(df, cov, 'm_b_corr', z_cutoff=0.023)

omegar = 8.24e-5

if __name__ == "__main__":

    def likelihood(theta):
        omegam, *theta = theta
        return logl_pantheon(omegam, omegar, theta)

    ns = run(
        likelihood,
        sys.argv[1],
        [UniformPrior(0.01, 0.99)],
        "ia",
        [("Omegam", r"\Omega_\mathrm{m}")],
        False,
    )
