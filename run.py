import sys
import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior
from flexknot import Prior
from anesthetic import make_2d_axes
from clustering import xmeans
from likelihood import desi_likelihood
from mpi4py import MPI

comm = MPI.COMM_WORLD


n = int(sys.argv[1])

paramnames = [
    (r"H0rd", r"H_0r_d"),
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


h0rd_prior = UniformPrior(3650, 18250)
omegam_prior = UniformPrior(0.01, 0.99)

flexknotprior = Prior(0, 1, -3, -0.01)


def prior(x):
    return np.concatenate([
        [h0rd_prior(x[0]), omegam_prior(x[1])],
        flexknotprior(x[2:])
    ])


def likelihood(theta):
    h0rd, omegam, *theta = theta
    omegar = 8.24e-5
    return desi_likelihood(h0rd, omegam, omegar, theta)


for i in range(10):
    x = np.random.rand(ndims)
    print(f"{likelihood(prior(x))=}")


file_root = f"distances_{n}"

if __name__ == "__main__":
    ns = pypolychord.run(likelihood, ndims, prior=prior,
                         nlive=nlive,
                         paramnames=paramnames,
                         file_root=file_root,
                         # cluster=xmeans,
                         read_resume=True)

    if comm.rank == 0:
        fig, axes = make_2d_axes(params[:2])
        ns.plot_2d(axes)
        fig.savefig(f"plots/{file_root}.pdf", bbox_inches='tight')
