import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from matplotlib import pyplot as plt
from matplotlib import gridspec
import smplotlib
from fgivenx import plot_lines, plot_dkl
from anesthetic import read_chains, make_2d_axes
from anesthetic.samples import merge_samples_weighted
from pypolychord.output import PolyChordOutput
from common import flexknotparamnames
from flexknot import FlexKnot
from alpha_plot import alpha_plot


def collect_chains(name, n, single=False, cobaya=False, dodgy_wcdm=False, zenodo=False):
    """Returns idx, ns, nss, pcs, prior"""
    if single:
        idx = [n]
    else:
        idx = range(1, n+1)
    if cobaya:
        nss = [read_chains(f"/home/ano23/dp/desi/chains/nonlinear_pk_v{i}/{name}/{name}_polychord_raw/{name}") for i in idx]
        pcs = [PolyChordOutput(f"/home/ano23/dp/desi/chains/nonlinear_pk_v{i}/{name}/{name}_polychord_raw", name) for i in idx]
    elif dodgy_wcdm:
        nss = [read_chains(f"chains/{name}_{i}" if i != 1 else f"chains/{name}_test_{i}") for i in idx]
        pcs = [PolyChordOutput("chains", f"{name}_{i}" if i != 1 else f"{name}_test_{i}")for i in idx]
    elif zenodo:
        nss = [read_chains(f"chains/{name}/{name}_{i}") for i in idx]
        pcs = [PolyChordOutput("chains", f"{name}/{name}_{i}") for i in idx]
    else:
        nss = [read_chains(f"chains/{name}_{i}") for i in idx]
        pcs = [PolyChordOutput("chains", f"{name}_{i}") for i in idx]
    if single:
        ns = nss[0]
        prior = ns.prior()
    else:
        prior = merge_samples_weighted([_ns.prior() for _ns in nss])
        # ns = merge_samples_weighted(nss)
        ns = merge_samples_weighted(nss, weights=[pc.logZ for pc in pcs])
    return idx, ns, nss, pcs, prior


def plot_samples_dkl(f, x, ns, prior, ax, color='C0', log=False, max_alpha=1.0,
                     **kwargs):
    fsamps = plot_lines(f, x, ns, weights=ns.get_weights(),
                        ax=ax[0], color=color)
    mean = np.mean(fsamps, axis=-1)
    sigma = np.std(fsamps, axis=-1)

    if "cache" in kwargs:
        kwargs['prior_cache'] = kwargs['cache'] + "_prior"
    dkl = plot_dkl(f, x, ns, prior_samples=prior,
                   weights=ns.get_weights(), ax=ax[1], color=color, **kwargs)
    alpha_plot(x, mean, sigma, ax[0], color, dkl,
               linecolor=color,
               max_alpha=max_alpha, **kwargs)
    ax[1].set_ylim(bottom=0)


def corner_plot(ns, cols, ax, color='C0'):
    if len(cols) > 1:
        _axes = ns.plot_2d(ax, label="posterior", color=color,
                           kinds=dict(
                               lower="kde_2d",
                               diagonal="hist_1d",
                               upper="scatter_2d"),
                           upper_kwargs=dict(ncompress=1000, alpha=0.1),
                           diagonal_kwargs=dict(alpha=0.5),
                           lower_kwargs=dict(alpha=0.5,
                                             levels=[0.99, 0.95, 0.68],
                                             nplot_2d=10_000))
        _axes.tick_params(labelsize='large')
        for ii in range(len(_axes)):
            for i in range(len(_axes)):
                _axes.iloc[ii, i].xaxis.get_offset_text().set_fontsize("large")
                _axes.iloc[ii, i].yaxis.get_offset_text().set_fontsize("large")
    else:
        ns[cols[0]].plot.hist(ax=ax, alpha=0.5, color=color,
                              bins=40, density=True)
        ax.set_xlabel(ns[cols].get_labels()[0])
        ax.tick_params()


def bayes_and_tension(name, n, idx, pcs, fig, ax, label=None, color='C0',
                      tension=False, cobaya=False, zenodo=False):
    ax[0].set_xlabel("$n$", fontsize='x-large')
    ax[0].set_ylabel(r"$\log{Z_n}$", fontsize='x-large')

    pclogZs = []
    pclogZerrs = []
    for i, pc in zip(idx, pcs):
        pclogZs.append(pc.logZ)
        pclogZerrs.append(pc.logZerr)
    pclogZs, pclogZerrs = np.array(pclogZs), np.array(pclogZerrs)
    try:
        if cobaya:
            lcdm = PolyChordOutput(f"/home/ano23/dp/desi/chains/nonlinear_pk_0/{name}/{name}_polychord_raw", name)
        else:
            lcdm = PolyChordOutput("chains", f"{name}/{name}_lcdm" if zenodo else f"{name}_lcdm")
        pclogZs -= lcdm.logZ
        pclogZerrs = np.sqrt(pclogZerrs**2 + lcdm.logZerr**2)
        ax[0].set_ylabel(r"$\log Z_n-\log Z_{\Lambda\text{CDM}}$",
                         fontsize='x-large')
        logR = logsumexp(pclogZs) - np.log(n)
        partials = np.e**(pclogZs - logsumexp(pclogZs))
        logRerr = (np.sum((partials * pclogZs)**2)
                   + lcdm.logZerr**2)**(0.5)
        ax[0].set_title("Bayes factors",
                        # f"\n$\\log Z_\\mathrm{{flexknot}}"
                        # f" - \\log Z_{{\\Lambda\\text{{CDM}}}}"
                        # f" = {logR:.2f} \\pm {logRerr:.2f}$"
                        # if "_" not in name else "Bayes factors",
                        fontsize='x-large')

        label = f"{label}\n($\\log Z = {logR:.2f} \\pm {logRerr:.2f}$)"
        ax[0].set_xticks(idx[4::5])
        ax[0].set_xticks(idx, minor=True)
    except FileNotFoundError:
        print("LCDM file not found :(")
    ax[0].errorbar(idx, pclogZs, yerr=pclogZerrs,
                   label=label,
                   marker='+', linestyle='None',
                   color=color)
    # plot tensions
    if tension:
        a, b = name.split('_')
        logRi = np.load(f"tensions/{name}_logRi.npy")
        nsa = read_chains(f"chains/{a}/{a}_lcdm")
        nsb = read_chains(f"chains/{b}/{b}_lcdm")
        nsab = read_chains(f"chains/{name}/{name}_lcdm")

        logRlcdm = nsab.stats(nsamples=1000).logZ - (
            nsa.stats(nsamples=1000).logZ + nsb.stats(nsamples=1000).logZ)

        ax[1].errorbar(idx, logRi.mean(axis=1), yerr=logRi.std(axis=1),
                       linestyle='None', marker='_', color=color)
        ax[1].axhline(logRlcdm.mean(), color=color, linestyle='--', label='LCDM')
        ax[1].text(n-2, logRlcdm.mean()+0.1, r'$\Lambda$CDM', color=color)
        ax[1].set_xlabel("$n$", fontsize='x-large')
        ax[1].set_ylabel(r"Tension $\log R_n$", fontsize='x-large')
        fig.align_ylabels(ax)


def plot(name, n, single, cobaya, fig=None, ax=None,
         color='C1', label=None, cols=None, dodgy_wcdm=False,
         tension=False, lower_a=0., zenodo=False, upper_w=0):
    params = flexknotparamnames(n, tex=False)

    idx, ns, nss, pcs, prior = collect_chains(name, n, single, cobaya,
                                              dodgy_wcdm, zenodo)
    ns_comp = ns.compress(1000)
    prior_comp = prior.compress(1000)

    if "rdrag" in ns and "H0" in ns:
        ns["H0rd"] = ns.rdrag * ns.H0
        ns.set_label("H0rd", r"$H_0r_\mathrm{d}$")
    if "omegam" in ns:
        ns = ns.rename(columns={"omegam": "Omegam"})

    if cols is None:
        cols = ['Omegam']
        for col in 'H0rd', 'H0':
            if col in ns:
                cols.append(col)

    if fig is None:
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(2, 2)

        # add 2x1 gridspec to top left
        top_left = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                    subplot_spec=gs[0, 0],
                                                    hspace=0.1,
                                                    height_ratios=[1, 0.5])
        if tension:
            top_right = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                         subplot_spec=gs[0, 1],
                                                         hspace=0.1,
                                                         height_ratios=[3, 2])
            ax_logR = fig.add_subplot(top_right[1])
        bottom_right = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                        subplot_spec=gs[1, 1],
                                                        hspace=0.1,
                                                        height_ratios=[1, 0.5])
        ax_dkl = fig.add_subplot(top_left[1])
        ax_zkl = fig.add_subplot(bottom_right[1])
        ax = [
            [
                fig.add_subplot(top_left[0], sharex=ax_dkl),
                ax_dkl,
            ],
            [
                fig.add_subplot(top_right[0], sharex=ax_logR),
                ax_logR,
            ] if tension else [fig.add_subplot(gs[0, 1])],
            make_2d_axes(cols, fig=fig, subplot_spec=gs[1, 0])[1]
            if len(cols) > 1 else
            fig.add_subplot(gs[1, 0]),
            [
                fig.add_subplot(bottom_right[0], sharex=ax_zkl),
                ax_zkl,
            ],
        ]
        plt.setp(ax[0][0].get_xticklabels(), visible=False)
        if tension:
            plt.setp(ax[1][0].get_xticklabels(), visible=False)
        plt.setp(ax[3][0].get_xticklabels(), visible=False)

    fk = FlexKnot(lower_a, 1)

    def f(x, theta):
        theta = theta[~np.isnan(theta)]
        return fk(x, theta)

    x = np.linspace(lower_a, 1, 100)
    plot_samples_dkl(
        f, x, ns_comp[params], prior_comp[params],
        ax[0], color=color, max_alpha=0.9,
    )
    ax[0][0].axhline(-1, color='k', linestyle='--')
    ax[0][0].set_xlabel("$a$", fontsize='x-large')
    ax[0][1].set_ylabel(r"$D_\mathrm{KL}(\mathcal{P}||\pi)$",
                        fontsize='x-large')
    ax[0][0].set_ylabel("$w(a)$", fontsize='x-large')
    ax[0][0].set(xlim=(0, 1), ylim=(-3, upper_w))
    ax[0][1].set_ylim(top=1.2)

    corner_plot(ns, cols, ax[2], color=color)
    for _ax in fig.axes:
        _ax.tick_params(labelsize='large')

    if not single:
        bayes_and_tension(name, n, idx, pcs, fig, ax[1], label=label,
                          color=color, tension=tension, cobaya=cobaya,
                          zenodo=zenodo)

        if label is not None:
            ax[1][0].legend(fontsize='medium', frameon=True, framealpha=0.5)
    else:
        from PIL import Image
        ax[1].imshow(np.asarray(Image.open("why_is_it_empty.png")),
                     origin='lower', extent=(0, 1, 0, 1))

    def fz(z, theta):
        theta = theta[~np.isnan(theta)]
        return fk(1/(1+z), theta)

    z = np.logspace(-3, np.log10(2.5), 100)
    plot_samples_dkl(fz, z, ns_comp[params], prior_comp[params], ax[3],
                     max_alpha=0.9,
                     color=color, log=True)
    ax[3][1].set_xlabel("$z$", fontsize='x-large')
    ax[3][0].set_ylabel("$w(z)$", fontsize='x-large')
    ax[3][0].set(xlim=(min(z), max(z)), ylim=(-3, upper_w), xscale='log')

    ax[3][1].set_ylabel(r"$D_\mathrm{KL}(\mathcal{P}||\pi)$",
                        fontsize='x-large')
    ax[3][1].set_ylim(top=1.2)
    ax[3][0].axhline(-1, color='k', linestyle='--')

    ax[0][0].set_title(r"$w(a)$ reconstruction", fontsize='x-large')
    ax[3][0].set_title(r"$w(z)$ reconstruction", fontsize='x-large')
    fig.align_ylabels(ax[0])
    fig.align_ylabels(ax[3])

    return fig, ax


colors = dict(
    desidr1='#58acbc',
    desidr2='#58acbc',
    pantheonplus='#d05a5c',
    desi_sdss='#867db8',
    desidr1_pantheonplus='#1f77b4',  # for desi+ia
    terminal_blue='#81A1C1',
    desidr1_des5y='#7B0043',
    desidr2_pantheonplus='#ff964f',
    desidr2_des5y='#caa0ff',
    desi='#58acbc',
    desi3='#58acbc',
    ia_h0='#d05a5c',
    desiia_h0='#1f77b4',  # for desi+ia
    desides5y='#7B0043',
    des5y='C1',
    # arsenal_red='#EF0107',
    desi3ia_h0='#ff964f',
    desi3des5y='#caa0ff',
)

titles = dict(
    desi="DESI",
    desi_sdss="DESI with SDSS LRGs",
    ia_h0="Pantheon+",
    des5y="DES5Y",
    desiia_h0="DESI + Pantheon+",
    desides5y="DESI + DES5Y",
    desi3ia_h0='DESI DR2 + Pantheon+',
    desi3des5y='DESI DR2 + DES5Y',
)


if __name__ == "__main__":
    plt.rcParams.update({
        'axes.labelsize': 'x-large',
    })
    name = sys.argv[1]
    n = int(sys.argv[2])
    try:
        single = 'i' == sys.argv[3]
        cobaya = 'cobaya' == sys.argv[3]
    except IndexError:
        single = False
        cobaya = False

    # fig, ax = plot(name, n, single, cobaya, color=colors.get(name, colors['desidr1']))
    # fig, ax = plot("desi", n, single, cobaya, color='k', label=r"DESI DR1")
    # fig, ax = plot("desi3", n, single, cobaya, fig, ax, color=colors['desidr2'], label=r"DESI DR2", dodgy_wcdm=True)
    # fig, ax = plot("desi3", n, single, cobaya, color=colors['desidr2'], label=r"DESI DR2 $w\in [-3, -0.01]$", dodgy_wcdm=True)
    # fig, ax = plot("desidr2_wide", n, single, cobaya, fig, ax, color=colors['desi3ia_h0'], label=r"DESI DR2 $w\in [-3, 1]$", upper_w=1)
    # fig.suptitle("DESI DR2 prior comparison", fontsize="xx-large")
    fig, ax = plot("desidr1_pantheonplus", n, single, cobaya, color=colors['desidr1_pantheonplus'], label="DESI DR1 + Pantheon+")
    fig, ax = plot("desidr2_pantheonplus", n, single, cobaya, fig, ax, color=colors['desidr2_pantheonplus'], label="DESI DR2 + Pantheon+")
    # fig, ax = plot("desidr1_des5y", n, single, cobaya, color=colors['desidr1_des5y'], label="DESI DR1 + DES5Y")
    # fig, ax = plot("desidr2_des5y", n, single, cobaya, fig, ax, color=colors['desidr2_des5y'], label="DESI DR2 + DES5Y")
    # cols = ["Omegam", "H0rd", "H0"]
    # #1f77b4
    # fig, ax = plot("desi", n, single, False, color='#58acbc', label="DESI", cols=cols)
    # fig, ax = plot("ia", n, single, False, fig, ax, color='#d05a5c', label="Pantheon+", cols=cols)
    # fig, ax = plot("desiia", n, single, False, fig, ax, color='C0', label="DESI & Pantheon+", cols=cols)
    fig.tight_layout()
    # fig.suptitle("DESI DR2 + Pantheon+ prior comparison", fontsize="xx-large")
    # ax[2].iloc[1, 2].remove()
    # ax[2].iloc[2, 1].remove()

    # fig.suptitle(f"{'cobaya_' if cobaya else ''}{name}_{n}{'_i' if single else ''}")
    plotpath = Path("plots") / name
    plotpath.mkdir(parents=True, exist_ok=True)
    # fig.savefig(plotpath / f"desi_vs_pantheon_{n}{'_i' if single else ''}.png",
    # fig.savefig(plotpath / f"{name}_{n}_cobaya_comparison_wa.pdf",
    fig.savefig(plotpath / f"{name}_DR1_comparison_{n}{'_i' if single else ''}_wa.pdf",
    # fig.savefig(plotpath / f"{name}_{n}{'_i' if single else ''}_wa.pdf",
                bbox_inches='tight')
    plt.show()
