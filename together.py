import sys
import numpy as np
from numpy import array
from likelihood import desi_likelihood
from ia import ia_likelihood
import pypolychord
from flexknot import Prior
from mpi4py import MPI
from anesthetic import make_2d_axes


comm = MPI.COMM_WORLD


n = int(sys.argv[1])
paramnames = [
    (r"H0rd", r"H_0r_d"),
    ("Mb", r"M_\mathrm{b}"),
    ("H0", r"H_0"),
    (r"Omegam", r"\Omega_\mathrm{m}"),
]

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
params = [paramname[0] for paramname in paramnames]


lower = array([3650, -20, 50, 0.01])
upper = array([18250, -18, 100, 0.99])
prior_range = upper - lower

flexknotprior = Prior(0, 1, -3, -0.01)


def prior(x):
    return np.concatenate([
        lower + x[:4] * prior_range,
        flexknotprior(x[4:])
    ])


def likelihood(theta):
    h0rd, Mb, h0, omegam, *theta = theta
    omegar = 8.24e-5
    return desi_likelihood(h0rd, omegam, omegar, theta) + ia_likelihood(Mb, h0, omegam, omegar, theta)



# for i in range(100):
#     print(i)
#     likelihood(prior(np.random.rand(ndims)))

file_root = f"together_{n}"
if __name__ == "__main__":
    ns = pypolychord.run(likelihood, ndims, prior=prior,
                         read_resume=True,
                         paramnames=paramnames,
                         file_root=file_root,
                         nlive=1000,
                         )

    params = [p[0] for p in paramnames]
    if comm.rank == 0:
        fig, ax = make_2d_axes(params[:5])
        ns.plot_2d(ax)
        fig.savefig(f"plots/{file_root}.pdf", bbox_inches='tight')
