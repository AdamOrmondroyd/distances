from numpy import concatenate, e, sqrt, log
from scipy.integrate import quad, cumulative_trapezoid
from scipy.constants import c
from flexknot import FlexKnot
from flexknot.utils import get_x_nodes_from_theta, get_y_nodes_from_theta


# c in units of km/s
c = c/1000

w = FlexKnot(0, 1)


def f_de(z, theta):
    a = 1 / (1 + z)
    ai = get_x_nodes_from_theta(theta, False)[::-1]
    wi = get_y_nodes_from_theta(theta, False)[::-1]
    # NOTE: "left" means the lower integration limit. a0 = 1
    ai = concatenate([[1], ai, [0]])
    left_a = ai[ai > a]  # aka ai
    limits = concatenate([left_a, [a]])
    right_a = ai[1:len(left_a)+1]  # aka ai+1
    left_w = wi[:len(left_a)]  # aka wi
    right_w = wi[1:len(right_a)+1]  # aka wi+1

    m = (left_w - right_w) / (left_a - right_a)
    one_plus_c = left_w - m * left_a + 1

    uppers = -one_plus_c * log(limits[1:]) - m * limits[1:]
    lowers = -one_plus_c * log(limits[:-1]) - m * limits[:-1]
    integrals = uppers - lowers

    result = e**(3*sum(integrals))
    return result


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


def dh_over_rs(z, h0rd, omegam, omegar, theta):
    return c / h0rd / h(z, omegam, omegar, theta)


def dm_over_rs(z, h0rd, omegam, omegar, theta):
    z_nodes = (1 / get_x_nodes_from_theta(theta, False) - 1)[::-1]
    z_nodes = concatenate([[0], z_nodes[z_nodes < z], [z]])
    return c / h0rd * sum([
        quad(lambda z: 1/h(z, omegam, omegar, theta), lower, upper)[0]
        for lower, upper in zip(z_nodes[:-1], z_nodes[1:])
    ])


def dv_over_rs(z, h0rd, omegam, omegar, theta):
    return (z * dm_over_rs(z, h0rd, omegam, omegar, theta) ** 2
            * dh_over_rs(z, h0rd, omegam, omegar, theta)) ** (1/3)


def dl(zhd, zhel, h0, omegam, omegar, theta):
    q0 = quad(lambda z: 1/h(z, omegam, omegar, theta), 0, zhd[0])[0]
    h_inverse = [1 / h(zi, omegam, omegar, theta) for zi in zhd]
    q = cumulative_trapezoid(h_inverse, zhd, initial=0) + q0
    return (1+zhel) * c / h0 * q


def h0_dl_over_c(zhd, zhel, omegam, omegar, theta):
    q0 = quad(lambda z: 1/h(z, omegam, omegar, theta), 0, zhd[0])[0]
    h_inverse = [1 / h(zi, omegam, omegar, theta) for zi in zhd]
    q = cumulative_trapezoid(h_inverse, zhd, initial=0) + q0
    return (1+zhel) * q
