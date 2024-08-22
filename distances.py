from numpy import concatenate, e, sqrt
from scipy.integrate import quad, cumulative_trapezoid
from scipy.constants import c
from flexknot import FlexKnot
from flexknot.utils import get_x_nodes_from_theta


# c in units of km/s
c = c/1000

w = FlexKnot(0, 1)


def f_de(z, theta):
    # split integral by flexknot nodes
    x_nodes = get_x_nodes_from_theta(theta, False)
    x_nodes = x_nodes[x_nodes > 1 / (1+z)]
    limits = concatenate([
        [0],
        (1 / x_nodes - 1)[::-1],
        [z],
    ])
    integral = sum([
        quad(lambda z: (1 + w(1/(1+z), theta)) / (1 + z), lower, upper)[0]
        for lower, upper in zip(limits[:-1], limits[1:])
    ])
    return e**(3*integral)


def h(z, omegam, omegar, theta):
    """
    H(z) / H0
    """
    return sqrt(
        omegam * (1 + z)**3
        + omegar * (1 + z)**4
        + (1 - omegam - omegar) * f_de(z, theta)
    )


# TODO: check units of everything - c is km/s!

def h0dh(z, omegam, omegar, theta):
    return c / h(z, omegam, omegar, theta)


def h0dm(z, omegam, omegar, theta):
    return c * quad(lambda z: 1/h(z, omegam, omegar, theta), 0, z)[0]


def dh_over_rs(z, h0rd, omegam, omegar, theta):
    return c / h0rd / h(z, omegam, omegar, theta)


def dm_over_rs(z, h0rd, omegam, omegar, theta):
    return c / h0rd * quad(lambda z: 1/h(z, omegam, omegar, theta), 0, z)[0]


def dv_over_rs(z, h0rd, omegam, omegar, theta):
    return (z * dm_over_rs(z, h0rd, omegam, omegar, theta) ** 2
            * dh_over_rs(z, h0rd, omegam, omegar, theta)) ** (1/3)


def dl_over_rs(z, h0rd, omegam, omegar, theta):
    return (1+z) * dm_over_rs(z, h0rd, omegam, omegar, theta)


def dl(z, h0, omegam, omegar, theta):
    q0 = quad(lambda z: 1/h(z, omegam, omegar, theta), 0, z[0])[0]
    h_inverse = [1 / h(zi, omegam, omegar, theta) for zi in z]
    q = cumulative_trapezoid(h_inverse, z, initial=0) + q0
    return (1+z) * c / h0 * q
