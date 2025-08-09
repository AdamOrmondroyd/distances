import numpy as np
import pypolychord
from flexknot import Prior
from pypolychord.priors import UniformPrior
from anesthetic import make_2d_axes
from mpi4py import MPI


comm = MPI.COMM_WORLD

flexknotprior = Prior(0, 1, -3, -0.01)
wideknotprior = Prior(0, 1, -3, 1)


def cplprior(x):
    # Need to return wn, w0
    # w0 = U[-3, 1]
    # wa = U[-3, 2]
    # wn = w0 + wa
    wa = -3 + 5 * x[0]
    w0 = -3 + 4 * x[1]
    wn = w0 + wa
    return [wn, w0]


def flexknotparamnames(n, tex=True):
    try:
        n = int(n)
    except ValueError:
        return []
    p = []
    if n >= 2:
        p += [("wn", "w_n")]

    for i in range(n-2, 0, -1):
        p += [
            (f"a{i}", f"a_{{{i}}}"),
            (f"w{i}", f"w_{{{i}}}"),
        ]
    if n >= 1:
        p += [("w0", "w_0")]
    if not tex:
        p = [pi[0] for pi in p]
    return p


def run(likelihood, n, priors,
        file_root, paramnames, read_resume,
        wide_prior=False, low_a=0):
    paramnames += flexknotparamnames(n)
    _flexknotprior = Prior(low_a, 1, -3, 1) if wide_prior else Prior(low_a, 1, -3, 0)
    try:
        n = int(n)

        def prior(x):
            return np.concatenate([
                [prior(xi) for xi, prior in zip(x, priors)],
                _flexknotprior(x[len(priors):])
            ])
    except ValueError:
        if n == 'cpl':
            print("it's CPLing time", flush=True)
            paramnames += [("wa", "w_a"), ("w0", "w_0")]

            def prior(x):
                return np.concatenate([
                    [prior(xi) for xi, prior in zip(x, priors)],
                    cplprior(x[len(priors):])
                ])

            # impose w0 + wa < 0
            _likelihood = likelihood

            def likelihood(theta):
                wa, w0 = theta[-2:]
                if w0 + wa > 0:
                    # print("returning log0 and moving on", flush=True)
                    return -np.inf
                return _likelihood(theta)
        else:
            def prior(x):
                return [prior(xi) for xi, prior in zip(x, priors)]

    ns = pypolychord.run(
        likelihood,
        len(paramnames),
        prior=prior,
        nlive=1000,
        nprior=10_000,
        paramnames=paramnames,
        base_dir=f"chains/{file_root}",
        file_root=f"{file_root}{'_wide' if wide_prior else ''}_{n}",
        read_resume=read_resume,
    )

    if comm.rank == 0:
        fig, axes = make_2d_axes([p[0] for p in paramnames[:len(priors)]])
        ns.plot_2d(axes)
        fig.savefig(f"plots/{file_root}.pdf", bbox_inches='tight')

    return ns
