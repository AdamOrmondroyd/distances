import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from matplotlib import pyplot as plt
from matplotlib import gridspec
import smplotlib
from fgivenx import plot_lines, plot_contours
from anesthetic import read_chains, make_2d_axes
from anesthetic.samples import merge_samples_weighted
from pypolychord.output import PolyChordOutput
from common import flexknotparamnames
from flexknot import FlexKnot

def collect_chains(name, n, single=False, cobaya=False):
    """Returns idx, ns, nss, pcs, prior"""
    if single:
        ns = read_chains(f"chains/{name}_{n}")
        prior = ns.prior()
        idx = [n]
    else:
        idx = range(1, n+1)
        if cobaya:
            nss = [read_chains(f"/home/ano23/dp/desi/chains/nonlinear_pk_v{i}/{name}/{name}_polychord_raw/{name}") for i in idx]
            pcs = [PolyChordOutput(f"/home/ano23/dp/desi/chains/nonlinear_pk_v{i}/{name}/{name}_polychord_raw", name) for i in idx]
        else:
            nss = [read_chains(f"chains/{name}_{i}") for i in idx]
            pcs = [PolyChordOutput("chains", f"{name}_{i}") for i in idx]
        prior = merge_samples_weighted([_ns.prior() for _ns in nss])
        # ns = merge_samples_weighted(nss)
        ns = merge_samples_weighted(nss, weights=[pc.logZ for pc in pcs])
    return idx, ns, nss, pcs, prior


def plot(name, n, single, cobaya, fig=None, ax=None, color='C1', label=None, cols=None):
    paramnames = flexknotparamnames(n)

    params = [p[0] for p in paramnames]

    idx, ns, nss, pcs, prior = collect_chains(name, n, single, cobaya)

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
        ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
              make_2d_axes(cols, fig=fig, subplot_spec=gs[1, 0])[1], fig.add_subplot(gs[1, 1])]

    # print(fig.get_children())
    # exit()
    prior_color='C0'

    fk = FlexKnot(0, 1)

    def f(x, theta):
        theta = theta[~np.isnan(theta)]
        return fk(x, theta)

    x = np.linspace(0, 1, 100)
    print(f"{ns[params]=}")
    # plot_lines(f, x, prior[params], weights=prior.get_weights(),
               # ax=ax[0], color=prior_color)
    plot_lines(f, x, ns[params], weights=ns.get_weights(),
               ax=ax[0], color=color)
    ax[0].set(xlabel="$a$", ylabel="$w(a)$",
              xlim=(0, 1), ylim=(-3, 0))

    # prior.plot_2d(ax[2], label="prior", color=prior_color, alpha=0.5,
                  # kinds=dict(lower="kde_2d", diagonal="hist_1d", upper="scatter_2d"))
    ns.plot_2d(ax[2], label="posterior", color=color,
               kinds=dict(lower="kde_2d", diagonal="hist_1d", upper="scatter_2d"),
               upper_kwargs=dict(ncompress=1000, alpha=0.1),
               diagonal_kwargs=dict(alpha=0.5),
               lower_kwargs=dict(alpha=0.5, levels=[0.99, 0.95, 0.68],
                                 nplot_2d=10_000))
    # ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    if not single:
        ax[1].set(xlabel="$n$", ylabel=r"$\log{Z_n}$")

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
                lcdm = PolyChordOutput("chains", f"{name}_lcdm")
            pclogZs -= lcdm.logZ
            pclogZerrs = np.sqrt(pclogZerrs**2 + lcdm.logZerr**2)
            ax[1].set(ylabel=r"$\log Z_n-\log Z_{\Lambda\text{CDM}}$")
            logR = logsumexp(pclogZs) - np.log(n)
            partials = np.e**(pclogZs - logsumexp(pclogZs))
            logRerr = (np.sum((partials * pclogZs)**2) + lcdm.logZerr**2)**(0.5)
            ax[1].set(title=f"$\\log Z_\\mathrm{{flexknot}} - \\log Z_{{\\Lambda\\text{{CDM}}}} = {logR:.2f} \\pm {logRerr:.2f}$")

            ax[1].set_xticks(idx[4::5])
            ax[1].set_xticks(idx, minor=True)
        except FileNotFoundError:
            print("LCDM file not found :(")
        ax[1].errorbar(idx, pclogZs, yerr=pclogZerrs,
                       label=label,
                       marker='+', linestyle='None',
                       color=color)
        if label is not None: ax[1].legend()

    def fz(z, theta):
        theta = theta[~np.isnan(theta)]
        return fk(1/(1+z), theta)

    z = np.logspace(-3, np.log10(2.5))
    # plot_lines(fz, z, prior[params], weights=prior.get_weights(),
               # ax=ax[3], color=prior_color)
    plot_lines(fz, z, ns[params], weights=ns.get_weights(),
               ax=ax[3], color=color)
    ax[3].set(xlabel="$z$", ylabel="$w(z)$",
              xlim=(min(z), max(z)), ylim=(-3, 0),
              xscale='log')
    return fig, ax


if __name__ == "__main__":
    name = sys.argv[1]
    n = int(sys.argv[2])
    try:
        single = 'i' == sys.argv[3]
        cobaya = 'cobaya' == sys.argv[3]
    except IndexError:
        single = False
        cobaya = False

    cols = ["Omegam", "H0rd", "H0"]
    fig, ax = plot("desi", n, single, False, color='C2', label="DESI", cols=cols)
    fig, ax = plot("ia", n, single, False, fig, ax, color='C1', label="Pantheon+", cols=cols)
    # fig, ax = plot(name, n, single, cobaya, color='C1')

    # fig.suptitle(f"{'cobaya_' if cobaya else ''}{name}_{n}{'_i' if single else ''}")
    fig.tight_layout()
    plotpath = Path("plots") / name
    plotpath.mkdir(parents=True, exist_ok=True)
    fig.savefig(plotpath / "desi_vs_pantheon.pdf", bbox_inches="tight")
    # fig.savefig(plotpath / f"{name}_{n}_cobaya_comparison_wa.pdf",
    # fig.savefig(plotpath / f"{name}_{n}{'_i' if single else ''}_wa.pdf",
                # bbox_inches='tight')
    plt.show()
