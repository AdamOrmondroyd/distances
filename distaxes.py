import jax
from jax import vmap
from jax import numpy as jnp
from jax.numpy import (
    exp, sqrt, log, concatenate, zeros, ones, searchsorted, linspace, trapezoid
)
from scipy.constants import c

# change to double precision
jax.config.update("jax_enable_x64", True)


c = c/1000


def integrate_cpl(ai, ai1, wi, wi1, alower=None):
    m = (wi - wi1) / (ai - ai1)
    oneplusc = wi - m * ai + 1
    if alower is None:
        alower = ai1
    return oneplusc * log(ai/alower) + m * (ai - alower)


def prep(z, a, w):
    # compute all but the last part of ∫1+w/1+zdz, and preemtively cumsum
    # Never need the last section precomputed as would need z=infinity
    a = concatenate([ones((1,) + a.shape[1:], dtype=a.dtype), a])
    section = integrate_cpl(a[:-1], a[1:], w[:-2], w[1:-1])
    # helpful to have zero at the beginning to handle
    # being in the zeroth flexknot section
    section = section.cumsum(axis=0)
    section = concatenate([zeros((1,) + section.shape[1:]), section])
    a = concatenate([a, zeros((1,) + a.shape[1:], dtype=a.dtype)])
    return a, section


def f_de(z, a, w, section):
    alower = 1/(1+z)
    i = vmap(lambda a: searchsorted(-a, -alower, side='right')-1, 1, 1)(a)

    ai = vmap(lambda col, idx: col[idx], in_axes=(1, 1), out_axes=1)(a, i)
    ai1 = vmap(lambda col, idx: col[idx+1], in_axes=(1, 1), out_axes=1)(a, i)
    wi = vmap(lambda col, idx: col[idx], in_axes=(1, 1), out_axes=1)(w, i)
    wi1 = vmap(lambda col, idx: col[idx+1], in_axes=(1, 1), out_axes=1)(w, i)
    section = vmap(lambda col, idx: col[idx], in_axes=(1, 1), out_axes=1)(section, i)

    # i = 2, 1000
    # a = 10, 1000
    # a[i] = 2, 1000, 1000 :(
    # treemap?

    return exp(3*(section + integrate_cpl(ai, ai1, wi, wi1, alower[:, None])))


def h(z, omegam, omegar, f_de):
    """
    H(z) / H0
    """
    return sqrt(
        omegam * (1 + z)**3
        + omegar * (1 + z)**4
        + (1 - omegam - omegar) * f_de
    )


def dh_over_rs(z, h0rd, h):
    return c / h0rd / h


def dm_over_rs(z, a, w, section, h0rd, omegam, omegar, f_de, resolution=100):
    _z = linspace(0, z, resolution)
    print(f"{_z.shape=}")
    _f_de = f_de(_z, a, w, section)
    print(f"{_f_de.shape=}")
    print(f"{omegam.shape=}")
    one_over_h = 1/h(_z[:, None, ...], omegam[:, :, None, ...], omegar, _f_de)
    print(f"{one_over_h.shape=}")
    return c / h0rd * trapezoid(one_over_h, _z[:, None, ...])



def logl(h0rd, omegam, omegar, theta=jnp.array([-1])):
    theta = jnp.array(theta)
    ai = jnp.concatenate([[1], theta[2:0:-2], [0]])
    wi = theta[::-2]


if __name__ == "__main__":
    import numpy as np
    from distances import f_de as f_de_old, dh_over_rs as dh_over_rs_old, dm_over_rs as dm_over_rs_old
    from flexknot.utils import create_theta
    from flexknot.priors import Prior
    from pypolychord.priors import UniformPrior

    h0rdprior = UniformPrior(3650, 18250)
    omprior = UniformPrior(0.01, 0.99)
    fkprior = Prior(0, 1, -3, 0)
    N = 5
    ndims = 2*N
    nsamples = 100

    def prior(x):
        return np.concatenate([h0rdprior(x[0:1]),  omprior(x[1:2]), fkprior(x[2:])])

    theta = jnp.array([prior(np.random.rand(ndims)) for _ in range(nsamples)]).T
    print(f"{theta.shape=}")
    h0rd = theta[0:1]
    omegam = theta[1:2]
    theta = theta[2:]
    a = theta[1:-1:2][::-1]
    w = jnp.vstack([theta[0::2], theta[-1]])[::-1]
    print(f"{a.shape=}")
    print(f"{w.shape=}")

    z = jnp.arange(0, 2, 0.01)

    a, sections = prep(z, a, w)
    print(f"{a=}")
    print(f"{a.shape=}")
    print(f"{sections=}")
    print(f"{sections.shape=}")

    new_f_de = f_de(z, a, w, sections)
    new_h = h(z[:, None, ...], omegam, 8.24e-5, new_f_de)
    print(f"{new_h}")
    print(f"{h(z[:, None, ...], omegam, 8.24e-5, new_f_de)=}")
    print(f"{new_f_de.shape=}")
    new = dh_over_rs(z, h0rd, new_h)
    old = np.array([[dh_over_rs_old(zi, h0rdi.squeeze(), omegami.squeeze(), 8.24e-5, theta_i) for zi in z] for h0rdi, omegami, theta_i in zip(h0rd.T, omegam.T, theta.T)])
    print(f"{old.shape=}")
    print(f"{new.shape=}")
    print(f"{old=}")
    print(f"{new=}")
    assert np.allclose(old.T, new)

    new_dm = dm_over_rs(z, a, w, sections, h0rd, omegam, 8.24e-5, f_de)
    old_dm = np.array([[dm_over_rs_old(zi, h0rdi, omegami, 8.24e-5, theta_i) for zi in z] for h0rdi, omegami, theta_i in zip(h0rd.T, omegam.T, theta.T)])
    print(f"{old_dm=}")
    print(f"{new_dm=}")
