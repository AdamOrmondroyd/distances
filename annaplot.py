from numpy import isnan, array, average, linspace
from matplotlib import pyplot as plt
import smplotlib
from flexknot import FlexKnot
from common import flexknotparamnames
from plot import collect_chains
from fgivenx import plot_lines


single = False
n = 20

_, desi, *_ = collect_chains("desidr1", n, single)
_, ia, *_ = collect_chains("pantheonplus", n, single)
_, des5y, *_ = collect_chains("des5y", n, single)
desicolor = '#58acbc'
iacolor = '#d05a5c'
des5y_color = 'C1'
params = flexknotparamnames(n, tex=False)

a = linspace(0, 1, 100)
fk = FlexKnot(0, 1)


def f(a, theta):
    theta = theta[~isnan(theta)]
    return fk(a, theta)


def average_f(a, samples, weights):
    return average(array([f(a, theta) for theta in samples]), weights=weights, axis=0)


fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for _ax in ax:
    _ax.axhline(-1, linestyle='--', color='k')

plot_lines(f, a, desi[params], weights=desi.get_weights(), ax=ax[0], color=desicolor)
ax[0].plot(a, average_f(a, desi[params].to_numpy(), desi.get_weights()), color=desicolor, linestyle='--', label="DESI")
ax[0].plot(a, average_f(a, ia[params].to_numpy(), ia.get_weights()), color=iacolor, linestyle='--', label="Pantheon+")
ax[0].plot(a, average_f(a, des5y[params].to_numpy(), des5y.get_weights()), color=des5y_color, linestyle='-.', label="DES5Y")

plot_lines(f, a, ia[params], weights=ia.get_weights(), ax=ax[1], color=iacolor)
ax[1].plot(a, average_f(a, ia[params].to_numpy(), ia.get_weights()), color=iacolor, linestyle='--')
ax[1].plot(a, average_f(a, desi[params].to_numpy(), desi.get_weights()), color=desicolor, linestyle='--')
ax[1].plot(a, average_f(a, des5y[params].to_numpy(), des5y.get_weights()), color=des5y_color, linestyle='-.')

plot_lines(f, a, des5y[params], weights=des5y.get_weights(), ax=ax[2], color=des5y_color)
ax[2].plot(a, average_f(a, ia[params].to_numpy(), ia.get_weights()), color=iacolor, linestyle='--')
ax[2].plot(a, average_f(a, desi[params].to_numpy(), desi.get_weights()), color=desicolor, linestyle='--')
ax[2].plot(a, average_f(a, des5y[params].to_numpy(), des5y.get_weights()), color=des5y_color, linestyle='-.')


for _ax, title in zip(ax, ["DESI", "Pantheon+", "DES5Y"]):
    _ax.set(xlim=(0, 1), ylim=(-3, 0),
            xlabel=r'$a$', ylabel=r'$w(a)$',
            title=title)

ax[0].legend(fontsize='small', loc='upper left')
fig.tight_layout()
fig.savefig('plots/comparison/annaplot.pdf', bbox_inches='tight')
plt.show()
