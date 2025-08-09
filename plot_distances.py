import sys
from functools import partial
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import read_chains
from fgivenx import plot_lines
from distances import dh_over_rs, dm_over_rs, dv_over_rs
from common import flexknotparamnames
from bao import dmdhplot, dvplot
import baodr2
import baosdss
from flexknot import FlexKnot
import smplotlib
from pypolychord.output import PolyChordOutput
from plot import collect_chains, plot_samples_dkl
from desidr2 import omegar


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


def plot_distances(name, n, single, color='C2', axs=None, dmdhgreys={}, dvgreys={}, desidr2=False, sdss=False, plot_data=True):
    idx, ns, nss, pcs, prior = collect_chains(name, n, single)

    np.random.seed(60022)
    ns = ns.compress(1000)
    prior = prior.compress(1000)

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
        lcdm_dm_over_zdh = np.array(dm_over_zdh(a, lcdm[params[:2]].to_numpy()[-1]))
        lcdm_dv_over_rdz23 = np.array(dv_over_rdz23(a, lcdm[params[:2]].to_numpy()[-1]))
        print(f"{lcdm_dm_over_zdh=}")
        plot_samples_dkl(
            lambda a, theta: dm_over_zdh(a, theta) - lcdm_dm_over_zdh,
            a, ns[params], prior[params], [axs[0], axs[1]], color,
            cache=f"cache/{name}_dm_over_zdh_lcdm",
            max_alpha=0.5,
        )
        plot_samples_dkl(
            lambda a, theta: dv_over_rdz23(a, theta) - lcdm_dv_over_rdz23,
            a, ns[params], prior[params], [axs[2], axs[3]], color,
            cache=f"cache/{name}_dv_over_rdz23_lcdm",
            max_alpha=0.5,
        )

    except FileNotFoundError:
        print("lcdm not found")

        plot_samples_dkl(dm_over_zdh, a, ns[params], prior[params], [axs[0], axs[1]],
                         color=color,
                         cache=f"cache/{name}_dm_over_zdh_{n}{'_i' if single else ''}")
        plot_samples_dkl(dv_over_rdz23, a, ns[params], prior[params], [axs[0], axs[1]],
                         color=color,
                         cache=f"cache/{name}_dv_over_rdz23_{n}{'_i' if single else ''}")


    # fsamps = plot_lines(fa, a, ns[params[2:]], weights=ns.get_weights(),
    #                     ax=axs[2], color=color)
    # # axs[2].plot(a, np.average(fsamps, axis=1, weights=ns.get_weights()), color=color, linestyle='--')
    # mean = np.mean(fsamps, axis=-1)
    # sigma = np.std(fsamps, axis=-1)
    # axs[2].plot(a, mean, color=color, linestyle='--')
    # axs[2].fill_between(a, mean-sigma, mean+sigma, color=color, alpha=0.5)

    if plot_data:
        if sdss:
            baosdss.dmdhplot(axs[0], partial(dm_over_zdh, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dmdhgreys)
            baosdss.dvplot(axs[2], partial(dv_over_rdz23, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dvgreys)
        elif desidr2:
            baodr2.dmdhplot(axs[0], partial(dm_over_zdh, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dmdhgreys)
            baodr2.dvplot(axs[2], partial(dv_over_rdz23, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dvgreys)
        else:
            dmdhplot(axs[0], partial(dm_over_zdh, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dmdhgreys)
            dvplot(axs[2], partial(dv_over_rdz23, theta=lcdm[params[:2]].to_numpy()[-1]), grey=dvgreys)
    axs[0].axhline(0, linestyle="--")
    axs[2].axhline(0, linestyle="--")
    axs[1].set(ylabel=r'$\mathcal{D}_\mathrm{KL}(\mathcal{P} || \pi)$')
    axs[3].set(xlabel='$a$', ylabel=r'$\mathcal{D}_\mathrm{KL}(\mathcal{P} || \pi)$')
    for ax in axs:
        ax.set(xlim=(0, 1))
    # axs[0].set(ylim=(17, 21.5))
    # axs[1].set(ylim=(0.95, 2.1))
    axs[0].set(ylim=(-0.25, 0.25))
    axs[1].set(ylim=(0, 5))
    axs[2].set(ylim=(-1.25, 1.25))
    axs[3].set(ylim=(0, 5))
    # axs[2].set(ylim=(-3, 0))


if __name__ == "__main__":
    name = sys.argv[1]
    n = int(sys.argv[2])
    try:
        single = 'i' == sys.argv[3]
    except IndexError:
        single = False

        # figsize=(22, 16)
    # fig, ax = plt.subplots(3, 5, figsize=(14, 10), sharex='col', sharey='row', gridspec_kw={"wspace": 0, "hspace": 0}) fig, ax = plt.subplots(3, 2, figsize=(8, 11), sharex='col', sharey='row', gridspec_kw={"wspace": 0, "hspace": 0})
    fig, ax = plt.subplots(4, 2, figsize=(8, 9), sharex='col', sharey='row',
                           # gridspec_kw={"wspace": 0, "hspace": 0},
                           height_ratios=[3, 1, 3, 1], constrained_layout=True)
    plot_distances("desidr1", n, single, 'k', ax[:, 0])
    plot_distances("desidr2", n, single, '#58acbc', ax[:, 1], desidr2=True)
    # plot_distances("desi_no_first_lrg", n, single, '#58acbc', ax[:, 1], dmdhgreys={0}, dvgreys={1})
    # plot_distances("desi_no_second_lrg", n, single, '#58acbc', ax[:, 2], dmdhgreys={1}, dvgreys={2})
    # plot_distances("desi_neither_lrg", n, single, '#58acbc', ax[:, 3], dmdhgreys={0, 1}, dvgreys={1, 2})
    # plot_distances("desi_sdss", n, single, '#867db8', ax[:, 4], sdss=True)
    for _ax in ax[:, 1:].flatten():
        _ax.set(ylabel="")
    for _ax in ax[:-1, :].flatten():
        _ax.set(xlabel="")
    for _ax in ax[-1, :-1].flatten():
        _ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        # _ax.get_xticklabels()[-1].set_visible(False)
    ax[-1, -1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # for _ax, title in zip(ax[0, :].flatten(), ["DESI", "remove $z=0.510$\n($a=0.662$)", "remove $z=0.706$\n($a=0.586$)", "remove both", "replace with\nSDSS LRGs"]):
        # _ax.set(title=title)
    for _ax, title in zip(ax[0, :].flatten(), ["DESI DR1", "DESI DR2"]):
        _ax.set(title=title)

    # ax[1, 0].legend(fontsize='x-small', loc='lower left')
    ax[2, -1].legend(fontsize='x-small', loc='lower left')
    fig.align_labels()
    # # fig, ax = plt.subplots(3, figsize=(6, 16), sharex='col', gridspec_kw={"wspace": 0, "hspace": 0})
    # # plot_distances(name, n, single, "#867db8", ax)
    # # ax[1].legend(fontsize='small', loc='lower left')
    # fig.tight_layout()

    plotpath = Path("plots") / name
    plotpath.mkdir(parents=True, exist_ok=True)
    fig.savefig(plotpath / f"{name}_{n}_distances{'_i' if single else ''}.pdf",
                bbox_inches='tight')
    plt.show()
