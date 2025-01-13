import numpy as np
import pypolychord
from flexknot import Prior
from clustering import xmeans
from anesthetic import make_2d_axes
from mpi4py import MPI


comm = MPI.COMM_WORLD

flexknotprior = Prior(0, 1, -3, -0.01)
flexknotprior_redshift = Prior(0, 3, -3, -0.01)


def flexknotparamnames(n, redshift=False, tex=True):
    try:
        n = int(n)
    except ValueError:
        return []
    p = []
    if n >= 2:
        p += [("wn", "w_n")]

    for i in range(n-2, 0, -1):
        p += [
            (f"z{i}", f"z_{i}") if redshift else (f"a{i}", f"a_{i}"),
            (f"w{i}", f"w_{i}"),
        ]
    if n >= 1:
        p += [("w0", "w_0")]
    if not tex:
        p = [pi[0] for pi in p]
    return p


def run(likelihood, n, priors,
        file_root, paramnames, read_resume, redshift=False):
    paramnames += flexknotparamnames(n, redshift)
    try:
        n = int(n)

        def prior(x):
            return np.concatenate([
                [prior(xi) for xi, prior in zip(x, priors)],
                flexknotprior_redshift(x[len(priors):]) if redshift else flexknotprior(x[len(priors):])
            ])
    except ValueError:
        def prior(x):
            return [prior(xi) for xi, prior in zip(x, priors)]

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
        fig, axes = make_2d_axes([p[0] for p in paramnames[:len(priors)]])
        ns.plot_2d(axes)
        fig.savefig(f"plots/{file_root}.pdf", bbox_inches='tight')

    return ns
