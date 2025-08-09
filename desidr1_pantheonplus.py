import sys
from common import run
from pypolychord.priors import GaussianPrior, UniformPrior
from desi import logl_desi
from pantheon import logl_pantheon


omegar = 8.24e-5


def likelihood(theta):
    h0rd, omegam, *theta = theta
    return logl_desi(h0rd, omegam, omegar, theta) + logl_pantheon(omegam, omegar, theta)


if __name__ == "__main__":
    ns = run(
        likelihood,
        sys.argv[1],
        [UniformPrior(3650, 18250),
         UniformPrior(0.01, 0.99)],
        "desidr1_pantheonplus",
        [(r"H0rd", r"H_0r_\mathrm{d}"),
         (r"Omegam", r"\Omega_\mathrm{m}")],
        True,
    )
