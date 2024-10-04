import numpy as np
import pypolychord
from flexknot import Prior
from cluster import xmeans
from anesthetic import make_2d_axes
from mpi4py import MPI


comm = MPI.COMM_WORLD

flexknotprior = Prior(0, 1, -3, -0.01)


def flexknotparamnames(n):
    try:
        n = int(n)
    except ValueError:
        return []
    p = []
    if n >= 2:
        p += [("wn", "w_n")]

    for i in range(n-2, 0, -1):
        p += [
            (f"a{i}", f"a_{i}"),
            (f"w{i}", f"w_{i}"),
        ]
    if n >= 1:
        p += [("w0", "w_0")]
    return p


def run(likelihood, n, prior_lower, prior_upper,
        file_root, paramnames, read_resume):
    paramnames += flexknotparamnames(n)
    prior_lower = np.array(prior_lower)
    prior_upper = np.array(prior_upper)
    prior_range = prior_upper - prior_lower
    try:
        n = int(n)

        def prior(x):
            return np.concatenate([
                prior_lower + x[:len(prior_range)] * prior_range,
                flexknotprior(x[len(prior_range):])
            ])
    except ValueError:
        def prior(x):
            return prior_lower + x * prior_range

    ns = pypolychord.run(
        likelihood,
        len(paramnames),
        prior=prior,
        nlive=1000,
        nprior=10_000,
        paramnames=paramnames,
        file_root=f"{file_root}_{n}",
        cluster=xmeans,
        read_resume=read_resume,
    )

    if comm.rank == 0:
        fig, axes = make_2d_axes([p[0] for p in paramnames[:len(prior_range)]])
        ns.plot_2d(axes)
        fig.savefig(f"plots/{file_root}.pdf", bbox_inches='tight')

    return ns
