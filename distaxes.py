import jax
from jax import numpy as jnp
from jax.numpy import (
    exp, sqrt, log, concatenate, zeros, ones, linspace, trapezoid,
    argmin, take_along_axis
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


def prep(a, w):
    # compute all but the last part of ∫1+w/1+zdz, and preemtively cumsum
    # Never need the last section precomputed as would need z=infinity
    a = concatenate([ones((1,) + a.shape[1:], dtype=a.dtype), a])
    sections = integrate_cpl(a[:-1], a[1:], w[:-2], w[1:-1])
    # helpful to have zero at the beginning to handle
    # being in the zeroth flexknot section
    sections = sections.cumsum(axis=0)
    sections = concatenate([zeros((1,) + sections.shape[1:]), sections])
    a = concatenate([a, zeros((1,) + a.shape[1:], dtype=a.dtype)])
    return a, sections


def f_de(z, a, w, sections):
    """
    z = (nbao, ...)
    a, w = (nfk, ...)
    result (nbao, ...)
    so nfk is "internal"
    """
    alower = 1/(1+z)
    i = argmin(jnp.where(a[:, None] > alower, a[:, None], jnp.inf), axis=0)
    # i should have shape (nbao, ...)
    ai = take_along_axis(a, i, axis=0)
    ai1 = take_along_axis(a, i+1, axis=0)
    wi = take_along_axis(w, i, axis=0)
    wi1 = take_along_axis(w, i+1, axis=0)
    section = take_along_axis(sections, i, axis=0)
    # ai should have shape (nbao, ...)

    return exp(3*(section + integrate_cpl(ai, ai1, wi, wi1, alower)))


def h(z, omegam, omegar, f_de):
    """
    H(z) / H0
    """
    return sqrt(
        omegam * (1 + z)**3
        + omegar * (1 + z)**4
        + (1 - omegam - omegar) * f_de
    )


@jax.jit
def dh_over_rs(z, a, w, sections, h0rd, omegam, omegar):
    _f_de = f_de(z, a, w, sections)
    _h = h(z, omegam, omegar, _f_de)
    return c / h0rd / _h


def dm_over_rs(z, a, w, sections, h0rd, omegam, omegar, resolution=1000):

    # _z = (nbao, resolution, ...)
    # should I sneak the additional axis behind all of them?
    _z = linspace(0, z, resolution, axis=-1)
    # the new axis needs to sneak behind the nbao and nfk axes
    _f_de = f_de(_z, a[..., None], w[..., None], sections[..., None])
    one_over_h = 1/h(_z, omegam[..., None], omegar, _f_de)
    return c / h0rd * trapezoid(one_over_h, _z, axis=-1)


def dv_over_rs(z, *args, **kwargs):
    return (z * dm_over_rs(z, *args, **kwargs) ** 2 * dh_over_rs(z, *args, **kwargs)) ** (1/3)
