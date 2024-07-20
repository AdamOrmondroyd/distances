import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior
from likelihood import likelihood
from flexknot import Prior

ndims = 3

paramnames = [
    (r"H0rd", r"H_0r_d"),
    (r"Omegam", r"\Omega_\mathrm{m}"),
    (r"w0", r"w_0")
]

h0rd_prior = UniformPrior(3650, 18250)
omegam_prior = UniformPrior(0.01, 0.99)

flexknotprior = Prior(0, 1, -3, -0.01)


def prior(x):
    return np.concatenate([
        [h0rd_prior(x[0]), omegam_prior(x[1])],
        flexknotprior(x[2:])
    ])


x = np.random.rand(ndims)

likelihood(prior(x))

pypolychord.run(likelihood, ndims, prior=prior,
                paramnames=paramnames)
