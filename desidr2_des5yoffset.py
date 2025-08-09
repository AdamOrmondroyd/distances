import sys
from common import run
from pypolychord.priors import GaussianPrior, UniformPrior
from desidr2 import logl_desi
from des5yoffset import logl_des5y


omegar = 8.24e-5


def likelihood(theta):
    delta_mb, h0rd, omegam, *theta = theta
    return logl_desi(h0rd, omegam, omegar, theta) + logl_des5y(delta_mb, omegam, omegar, theta)


if __name__ == "__main__":
    ns = run(
        likelihood,
        sys.argv[1],
        [
            UniformPrior(-0.1, 0.1),
            UniformPrior(3650, 18250),
            UniformPrior(0.01, 0.99)
        ],
        "desidr2_des5y",
        [
            ("delta_mb", r"\Delta m_\mathrm{B}"),
            (r"H0rd", r"H_0r_\mathrm{d}"),
            (r"Omegam", r"\Omega_\mathrm{m}")
         ],
        read_resume=True,
        wide_prior=True,
    )
