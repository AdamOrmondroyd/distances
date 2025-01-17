from numpy import concatenate, e, sqrt, log
from scipy.integrate import quad, cumulative_trapezoid
from scipy.constants import c
from flexknot import FlexKnot
from flexknot.utils import get_x_nodes_from_theta, get_y_nodes_from_theta


# c in units of km/s
c = c/1000

w = FlexKnot(0, 1)
wz = FlexKnot(0, 3)


def f_de_redshift(z, theta):
    x_nodes = get_x_nodes_from_theta(theta, False)
    x_nodes = x_nodes[x_nodes < z]
    limits = concatenate([[0], x_nodes, [z]])
    integral = sum([
        quad(lambda z: (1 + wz(z, theta)) / (1 + z), lower, upper)[0]
        for lower, upper in zip(limits[:-1], limits[1:])
    ])
    return e**(3*integral)


def f_de_redshift_analytic(z, theta):
    # WARN: I think the right_x_nodes are wrong
    x_nodes = get_x_nodes_from_theta(theta, False)
    y_nodes = get_y_nodes_from_theta(theta, False)
    left_x_nodes = concatenate([[0], x_nodes[x_nodes < z]])
    limits = concatenate([left_x_nodes, [z]])
    right_x_nodes = x_nodes[1:len(left_x_nodes)+1]
    left_y_nodes = y_nodes[:len(left_x_nodes)]
    right_y_nodes = y_nodes[1:len(left_x_nodes)+1]
    m = (right_y_nodes - left_y_nodes) / (right_x_nodes - left_x_nodes)
    c = left_y_nodes - m * left_x_nodes

    one_plus_c_minus_m = 1 + c - m
    integrals = (
        one_plus_c_minus_m * log(1 + limits[1:]) + m * limits[1:]
        - one_plus_c_minus_m * log(1 + limits[:-1]) - m * limits[:-1]
    )
    return e**(3*sum(integrals))


def f_de_analytic(z, theta):
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


def h(z, omegam, omegar, theta, redshift=False):
    """
    H(z) / H0
    """
    if redshift:
        return sqrt(
            omegam * (1 + z)**3
            + omegar * (1 + z)**4
            + (1 - omegam - omegar) * f_de_redshift_analytic(z, theta)
        )
    return sqrt(
        omegam * (1 + z)**3
        + omegar * (1 + z)**4
        + (1 - omegam - omegar) * f_de_analytic(z, theta)
    )


# TODO: check units of everything - c is km/s!

def h0dh(z, omegam, omegar, theta, redshift=False):
    return c / h(z, omegam, omegar, theta, redshift)


def h0dm(z, omegam, omegar, theta, redshift=False):
    return c * quad(lambda z: 1/h(z, omegam, omegar, theta, redshift), 0, z)[0]


def dh_over_rs(z, h0rd, omegam, omegar, theta, redshift=False):
    return c / h0rd / h(z, omegam, omegar, theta, redshift)


def dm_over_rs(z, h0rd, omegam, omegar, theta, redshift=False):
    return c / h0rd * quad(lambda z: 1/h(z, omegam, omegar, theta, redshift), 0, z)[0]


def dv_over_rs(z, h0rd, omegam, omegar, theta, redshift=False):
    return (z * dm_over_rs(z, h0rd, omegam, omegar, theta, redshift) ** 2
            * dh_over_rs(z, h0rd, omegam, omegar, theta, redshift)) ** (1/3)


def dl_over_rs(zhd, zhel, h0rd, omegam, omegar, theta, redshift=False):
    return (1+zhel) * dm_over_rs(zhd, h0rd, omegam, omegar, theta, redshift)


def dl(zhd, zhel, h0, omegam, omegar, theta, redshift=False):
    q0 = quad(lambda z: 1/h(z, omegam, omegar, theta, redshift), 0, zhd[0])[0]
    h_inverse = [1 / h(zi, omegam, omegar, theta, redshift) for zi in zhd]
    q = cumulative_trapezoid(h_inverse, zhd, initial=0) + q0
    return (1+zhel) * c / h0 * q
