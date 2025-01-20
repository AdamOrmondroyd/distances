import sys
from common import run
from pypolychord.priors import GaussianPrior, UniformPrior
from desi import logl_desi
from ia import logl_ia


omegar = 8.24e-5


def likelihood(theta):
    h0rd, h0, omegam, *theta = theta
    return logl_desi(h0rd, omegam, omegar, theta) + logl_ia(h0, omegam, omegar, theta)


if __name__ == "__main__":
    ns = run(
        likelihood,
        sys.argv[1],
        [UniformPrior(3650, 18250),
         UniformPrior(20, 100),
         UniformPrior(0.01, 0.99)],
        "desiia",
        [(r"H0rd", r"H_0r_\mathrm{d}"),
         ("H0", r"H_0"),
         (r"Omegam", r"\Omega_\mathrm{m}")],
        True,
    )
