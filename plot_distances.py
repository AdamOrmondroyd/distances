import sys
from functools import partial
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted
from fgivenx import plot_lines
from distances import dh_over_rs, dm_over_rs, dv_over_rs
from common import flexknotparamnames
from bao import dmdhplot, dvplot
import baosdss
from flexknot import FlexKnot
import smplotlib
from pypolychord.output import PolyChordOutput
from plot import collect_chains
from desi import omegar


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


def plot_distances(name, n, single, color='C2', axs=None, dmdhgreys={}, dvgreys={}, sdss=False):
    idx, ns, nss, pcs, prior = collect_chains(name, n, single)

    ns = ns.compress()

    # H0rd, Omegam, flexknot

    params = [
        "H0rd",
        "Omegam",
    ]

    params += flexknotparamnames(n, tex=False)

    print(ns)
    print(f"{ns.columns=}")
    print(f"{params=}")

    if axs is None:
        fig, axs = plt.subplots(3, figsize=(6, 14))
    # z = np.linspace(0.01, 2.5)
    a = np.linspace(0.001, 0.999)
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
        fsamps = plot_lines(lambda a, theta: dm_over_zdh(a, theta) - lcdm_dm_over_zdh,
                   a, ns[params], weights=ns.get_weights(), ax=axs[0],
                   cache=f"cache/{name}_dm_over_zdh_{n}{'_i' if single else ''}",
                   color=color,
                   parallel=True)
        mean = np.mean(fsamps, axis=-1)
        sigma = np.std(fsamps, axis=-1)
        axs[0].plot(a, mean, color=color, linestyle='--')
        axs[0].fill_between(a, mean-sigma, mean+sigma, color=color, alpha=0.5)

        fsamps = plot_lines(lambda a, theta: dv_over_rdz23(a, theta) - lcdm_dv_over_rdz23,
                   a, ns[params], weights=ns.get_weights(), ax=axs[1],
                   cache=f"cache/{name}_dv_over_rdz23_{n}{'_i' if single else ''}",
                   color=color,
                   parallel=True)
        mean = np.mean(fsamps, axis=-1)
        sigma = np.std(fsamps, axis=-1)
        axs[1].plot(a, mean, color=color, linestyle='--')
        axs[1].fill_between(a, mean-sigma, mean+sigma, color=color, alpha=0.5)

    except FileNotFoundError:
        print("lcdm not found")

        plot_lines(dm_over_zdh, a, ns[params], weights=ns.get_weights(), ax=axs[0],
                   cache=f"cache/{name}_dm_over_zdh_{n}{'_i' if single else ''}",
                   color=color,
                   parallel=True)
        plot_lines(dv_over_rdz23, a, ns[params], weights=ns.get_weights(), ax=axs[1],
                   cache=f"cache/{name}_dv_over_rdz23_{n}{'_i' if single else ''}",
                   color=color,
                   parallel=True)
        axs[2].set(ylim=(-0.25, 0.25))

    fk = FlexKnot(0, 1)


    def fa(a, theta):
        theta = theta[~np.isnan(theta)]
        return fk(a, theta)


    fsamps = plot_lines(fa, a, ns[params[2:]], weights=ns.get_weights(),
                        ax=axs[2], color=color)
    # axs[2].plot(a, np.average(fsamps, axis=1, weights=ns.get_weights()), color=color, linestyle='--')
    mean = np.mean(fsamps, axis=-1)
    sigma = np.std(fsamps, axis=-1)
    axs[2].plot(a, mean, color=color, linestyle='--')
    axs[2].fill_between(a, mean-sigma, mean+sigma, color=color, alpha=0.5)

    if sdss:
        baosdss.dmdhplot(axs[0], partial(dm_over_zdh, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dmdhgreys)
        baosdss.dvplot(axs[1], partial(dv_over_rdz23, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dvgreys)
    else:
        dmdhplot(axs[0], partial(dm_over_zdh, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dmdhgreys)
        dvplot(axs[1], partial(dv_over_rdz23, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dvgreys)
    axs[0].axhline(0, linestyle="--")
    axs[1].axhline(0, linestyle="--")
    axs[2].axhline(-1, linestyle="--")
    axs[2].set(xlabel='$a$', ylabel='$w(a)$')
    for ax in axs:
        ax.set(xlim=(0, 1))
    # axs[0].set(ylim=(17, 21.5))
    # axs[1].set(ylim=(0.95, 2.1))
    axs[0].set(ylim=(-0.25, 0.25))
    axs[1].set(ylim=(-1.25, 1.25))
    axs[2].set(ylim=(-3, 0))


if __name__ == "__main__":
    name = sys.argv[1]
    n = int(sys.argv[2])
    try:
        single = 'i' == sys.argv[3]
    except IndexError:
        single = False

    fig, ax = plt.subplots(3, 5, figsize=(22, 16), sharex='col', sharey='row', gridspec_kw={"wspace": 0, "hspace": 0})
    plot_distances("desi", n, single, '#58acbc', ax[:, 0])
    plot_distances("desi_no_first_lrg", n, single, '#58acbc', ax[:, 1], dmdhgreys={0}, dvgreys={1})
    plot_distances("desi_no_second_lrg", n, single, '#58acbc', ax[:, 2], dmdhgreys={1}, dvgreys={2})
    plot_distances("desi_neither_lrg", n, single, '#58acbc', ax[:, 3], dmdhgreys={0, 1}, dvgreys={1, 2})
    plot_distances("desi_sdss", n, single, '#867db8', ax[:, 4], sdss=True)
    for _ax in ax[:, 1:].flatten():
        _ax.set(ylabel="")
    for _ax in ax[:-1, :].flatten():
        _ax.set(xlabel="")
    for _ax in ax[-1, :-1].flatten():
        _ax.get_xticklabels()[-1].set_visible(False)
    for _ax, title in zip(ax[0, :].flatten(), ["DESI", r"remove $z=0.510$ ($a=0.662$)", r"remove $z=0.706$ ($a=0.586$)", "remove both", "replace with SDSS LRGs"]):
        _ax.set(title=title)

    ax[1, 0].legend(fontsize='small', loc='lower left')
    ax[1, -1].legend(fontsize='small', loc='lower left')
    # fig, ax = plt.subplots(3, figsize=(6, 16), sharex='col', gridspec_kw={"wspace": 0, "hspace": 0})
    # plot_distances(name, n, single, "#867db8", ax)
    # ax[1].legend(fontsize='small', loc='lower left')
    fig.tight_layout()

    plotpath = Path("plots") / name
    plotpath.mkdir(parents=True, exist_ok=True)
    fig.savefig(plotpath / f"{name}_{n}_distances{'_i' if single else ''}.pdf",
                bbox_inches='tight')
    plt.show()
