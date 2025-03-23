from pypolychord.priors import UniformPrior, SortedUniformPrior
from flexknot.utils import create_theta
from flexknot.priors import Prior
import distances
import distaxes
import numpy as np
import jax
import jax.numpy as jnp

# change to double precision
jax.config.update("jax_enable_x64", True)

omegar = 8.24e-5
h0rdprior = UniformPrior(3650, 18250)
omprior = UniformPrior(0.01, 0.99)
fkprior = Prior(0, 1, -3, 0)
wprior = UniformPrior(-3, 0)
def aprior(x): return SortedUniformPrior(0, 1)(x)[::-1]


def test_dh():
    N = 5
    ndims = 2*N
    x = jnp.array(np.random.rand(ndims))
    h0rd = h0rdprior(x[0])
    omegam = omprior(x[1])
    a = aprior(x[2:N])
    w = wprior(x[N:])
    theta = create_theta(a[::-1], w[::-1])
    z = jnp.array(0.510)
    a, sections = distaxes.prep(a, w)

    assert np.allclose(distances.dh_over_rs(z, h0rd, omegam, omegar, theta),
                       distaxes.dh_over_rs(z, a, w, sections, h0rd, omegam, omegar))
    z = jnp.arange(0.01, 2, 0.01)
    assert np.allclose(distaxes.dh_over_rs(z, a, w, sections, h0rd, omegam, omegar),
                      [distances.dh_over_rs(zi, h0rd, omegam, omegar, theta) for zi in z])


def test_dh_vectorized():
    N = 5
    ndims = 2*N
    nsamples = 100
    x = jnp.array(np.random.rand(ndims, nsamples))
    h0rd = h0rdprior(x[0])
    omegam = omprior(x[1])
    a = jnp.array([aprior(xi) for xi in x[2:N].T]).T
    w = wprior(x[N:])
    theta = np.array([create_theta(ai[::-1], wi[::-1]) for ai, wi in zip(a.T, w.T)]).T
    z = jnp.array(0.510)
    a, sections = distaxes.prep(a, w)

    assert np.allclose([distances.dh_over_rs(z, h0rdi, omegami, omegar, thetai)
                        for h0rdi, omegami, thetai in zip(h0rd, omegam, theta.T)],
                       distaxes.dh_over_rs(z, a, w, sections, h0rd, omegam, omegar))

    z = jnp.arange(0.01, 2, 0.01)
    assert np.allclose([[distances.dh_over_rs(zi, h0rdi, omegami, omegar, thetai)
                        for h0rdi, omegami, thetai in zip(h0rd, omegam, theta.T)]
                        for zi in z],
                       distaxes.dh_over_rs(z[..., None], a, w, sections, h0rd, omegam, omegar))
