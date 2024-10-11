import sys
from common import run
from pypolychord.priors import GaussianPrior, UniformPrior
from desi import logl_desi
from ia import logl_ia


omegar = 8.24e-5


def likelihood(theta):
    rd, h0, omegam, *theta = theta
    h0rd = h0 * rd
    return logl_desi(h0rd, omegam, omegar, theta) + logl_ia(h0, omegam, omegar, theta)


if __name__ == "__main__":
    ns = run(
        likelihood,
        sys.argv[1],
        [GaussianPrior(147.09, 0.26),
         UniformPrior(20, 100),
         UniformPrior(0.01, 0.99)],
        "desiia",
        [(r"H0rd", r"H_0r_d"),
         ("H0", r"H_0"),
         (r"Omegam", r"\Omega_\mathrm{m}")],
        True,
    )
