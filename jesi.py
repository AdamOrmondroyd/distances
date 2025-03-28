import sys
import numpy as np
import jax.numpy as jnp
from pandas import read_csv
from pypolychord.priors import UniformPrior
from common import run
import distaxes
from flexknot.utils import get_x_nodes_from_theta, get_y_nodes_from_theta
import jax

data = read_csv("desi3/desi3_mean.txt", header=None, index_col=None, sep=r"\s+", comment="#")
cov = np.loadtxt("desi3/desi3_cov.txt")

zs = jnp.array(data.iloc[:, 0].to_numpy())
mean = data.iloc[:, 1].to_numpy()
di_over_rss = [getattr(distaxes, i.lower()) for i in data.iloc[:, 2]]

invcov_over_2 = jnp.linalg.inv(cov) / 2
lognorm = -jnp.linalg.slogdet(2*jnp.pi*cov)[1] / 2


def logl_desi(h0rd, omegam, omegar, theta=np.array([-1])):
    theta = jnp.array(theta)
    a = get_x_nodes_from_theta(theta, False)[::-1]
    w = get_y_nodes_from_theta(theta, False)[::-1]
    params = {
        "h0rd": h0rd,
        "omegam": omegam,
        "omegar": omegar,
        "a": a,
        "w": w,
    }
    return logl_jax(params)


@jax.jit
def logl_jax(params):
    h0rd = params['h0rd']
    omegam = params['omegam']
    # omegar = params['omegar']
    a = params['a']
    w = params['w']

    a, sections = distaxes.prep(a, w)
    x = jnp.array([
        di_over_rs(z, a, w, sections, h0rd, omegam, omegar)
        for z, di_over_rs in zip(zs, di_over_rss)
    ]).squeeze()

    y = x - mean
    new = -y[None, :] @ invcov_over_2 @ y[:, None] + lognorm
    return new[..., 0, 0]


omegar = 8.24e-5

if __name__ == "__main__":

    def likelihood(theta):
        h0rd, omegam, *theta = theta

        return float(logl_desi(h0rd, omegam, omegar, theta))

    ns = run(
        likelihood,
        sys.argv[1],
        [UniformPrior(3650, 18250), UniformPrior(0.01, 0.99)],
        "jesi",
        [(r"H0rd", r"H_0r_\mathrm{d}"), (r"Omegam", r"\Omega_\mathrm{m}")],
        False,
    )
