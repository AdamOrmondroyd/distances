from numpy import e, sqrt
from scipy.integrate import quad
import flexknot

w = flexknot.FlexKnot(0, 1)


def integrand(z, theta):
    return (1 + w(1/(1+z), theta)) / (1 + z)


def f_de(z, theta):
    integral = quad(lambda z: (1 + w(1/(1+z), theta)) / (1 + z), 0, z)
    return e**(3*integral)


def h(z, omegam, omegar, theta):
    """
    H(z) / H0
    """
    return sqrt(
        omegam / (1 + z)**3
        + omegar / (1 + z)**4
        + (1 - omegam - omegar) * f_de(z, theta)
    )

def dv_over_rs(z, theta):
    return 0


def dm_over_rs(z, theta):
    return 0


def dh_over_rs(z, theta):
    return 0
