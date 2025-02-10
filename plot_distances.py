import sys
from functools import partial
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted
from fgivenx import plot_lines
from distances import dh_over_rs, dm_over_rs, dv_over_rs
from bao import dmdhplot, dvplot
from flexknot import FlexKnot
import smplotlib


name = sys.argv[1]
n = int(sys.argv[2])
try:
    single = 'i' == sys.argv[3]
except IndexError:
    single = False

if single:
    ns = read_chains(f"chains/{name}_{n}")
else:
    idx = range(1, n+1)
    nss = [read_chains(f"chains/{name}_{n}") for i in idx]
    ns = merge_samples_weighted(nss)

ns = ns.compress()

# H0rd, Omegam, flexknot

omegar = 8.24e-5

params = [
    "H0rd",
    "Omegam",
]

if n >= 2:
    params += ["wn"]

for i in range(n-2, 0, -1):
    params += [
        f"a{i}",
        f"w{i}",
    ]
if n >= 1:
    params += ["w0"]

print(ns)
print(f"{ns.columns=}")
print(f"{params=}")


def dm_over_zdh(a, theta):
    z = 1/a - 1
    h0rd, omegam, *theta = theta
    theta = np.array(theta)
    theta = theta[~np.isnan(theta)]

    return [
        dm_over_rs(zi, h0rd, omegam, omegar, theta)
        / (zi * dh_over_rs(zi, h0rd, omegam, omegar, theta))
        for zi in z
    ]


def dv_over_rdz23(a, theta):
    z = 1/a - 1
    h0rd, omegam, *theta = theta
    theta = np.array(theta)
    theta = theta[~np.isnan(theta)]

    return [
        dv_over_rs(zi, h0rd, omegam, omegar, theta) / zi**(2/3) for zi in z
    ]


fig, axs = plt.subplots(3, figsize=(6, 14))
# z = np.linspace(0.01, 2.5)
a = np.linspace(0, 1)
try:
    lcdm = read_chains(f"chains/{name}_lcdm")
    # plot_lines(dm_over_zdh, a, lcdm[params[:2]], weights=lcdm.get_weights(),
    # plot_lines(lambda a, theta: dm_over_zdh(a, theta) - lcdm_dm_over_zdh,
    #            color='C5',
    #            ax=axs[0],
    #            cache=f"cache/{name}_dm_over_zdh_lcdm",
    #            parallel=True)
    # plot_lines(dv_over_rdz23, a, lcdm[params[:2]], weights=lcdm.get_weights(),
    #            color='C5',
    #            ax=axs[1],
    #            cache=f"cache/{name}_dv_over_rdz23_lcdm",
    #            parallel=True)
    lcdm_dm_over_zdh = np.array(dm_over_zdh(a, lcdm[params[:2]].to_numpy()[-1]))
    lcdm_dv_over_rdz23 = np.array(dv_over_rdz23(a, lcdm[params[:2]].to_numpy()[-1]))
    print(f"{lcdm_dm_over_zdh=}")
    plot_lines(lambda a, theta: dm_over_zdh(a, theta) - lcdm_dm_over_zdh,
               a, ns[params], weights=ns.get_weights(), ax=axs[0],
               cache=f"cache/{name}_dm_over_zdh_{n}{'_i' if single else ''}",
               color='C2',
               parallel=True)
    plot_lines(lambda a, theta: dv_over_rdz23(a, theta) - lcdm_dv_over_rdz23,
               a, ns[params], weights=ns.get_weights(), ax=axs[1],
               cache=f"cache/{name}_dv_over_rdz23_{n}{'_i' if single else ''}",
               color='C2',
               parallel=True)

except FileNotFoundError:
    print("lcdm not found")

    plot_lines(dm_over_zdh, a, ns[params], weights=ns.get_weights(), ax=axs[0],
               cache=f"cache/{name}_dm_over_zdh_{n}{'_i' if single else ''}",
               color='C2',
               parallel=True)
    plot_lines(dv_over_rdz23, a, ns[params], weights=ns.get_weights(), ax=axs[1],
               cache=f"cache/{name}_dv_over_rdz23_{n}{'_i' if single else ''}",
               color='C2',
               parallel=True)

fk = FlexKnot(0, 1)


def fa(a, theta):
    theta = theta[~np.isnan(theta)]
    return fk(a, theta)


plot_lines(fa, a, ns[params[2:]], weights=ns.get_weights(),
           ax=axs[2], color='C2')

dmdhplot(axs[0], partial(dm_over_zdh, theta=lcdm[params[:2]].to_numpy()[-1]))
dvplot(axs[1], partial(dv_over_rdz23, theta=lcdm[params[:2]].to_numpy()[-1]))
axs[1].legend()
axs[0].axhline(0, linestyle="--")
axs[1].axhline(0, linestyle="--")
axs[2].axhline(-1, linestyle="--")
axs[2].set(xlabel='$a$', ylabel='$w(a)$')
for ax in axs:
    ax.set(xlim=(0, 1))
# axs[0].set(ylim=(17, 21.5))
# axs[1].set(ylim=(0.95, 2.1))
axs[2].set(ylim=(-3, 0))


fig.tight_layout()

plotpath = Path("plots") / name
plotpath.mkdir(parents=True, exist_ok=True)
fig.savefig(plotpath / f"{name}_{n}_distances{'_i' if single else ''}.pdf",
            bbox_inches='tight')
plt.show()
