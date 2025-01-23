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

if __name__ == "__main__":
    name = sys.argv[1]
    n = int(sys.argv[2])
    try:
        single = 'i' == sys.argv[3]
        cobaya = 'cobaya' == sys.argv[3]
    except IndexError:
        single = False
        cobaya = False
else:
    name = "distances"
    n = 9
    single = False

paramnames = flexknotparamnames(n)

params = [p[0] for p in paramnames]

if single:
    ns = read_chains(f"chains/{name}_{n}")
    prior = ns.prior()
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

if "rdrag" in ns and "H0" in ns:
    ns["H0rd"] = ns.rdrag * ns.H0
    ns.set_label("H0rd", r"$H_0r_\mathrm{d}$")

if __name__ == "__main__":
    prior_color = 'C0'
    post_color = 'C1'
    cols = ['omegam' if cobaya else 'Omegam', 'H0rd' if 'H0rd' in ns else 'H0']
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)
    ax = [plt.subplot(gs[0, 0]), None,
          plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1])]
    fig, axac = make_2d_axes(cols, fig=fig, subplot_spec=gs[1, 0])

    fk = FlexKnot(0, 1)

    def f(x, theta):
        theta = theta[~np.isnan(theta)]
        return fk(x, theta)

    x = np.linspace(0, 1, 100)
    print(f"{ns[params]=}")
    plot_lines(f, x, prior[params], weights=prior.get_weights(),
               ax=ax[0], color=prior_color)
    plot_lines(f, x, ns[params], weights=ns.get_weights(),
               ax=ax[0], color=post_color)
    ax[0].set(xlabel="$a$", ylabel="$w(a)$",
              xlim=(0, 1), ylim=(-3, 0))

    prior.plot_2d(axac, label="prior", color=prior_color,
                  kinds=dict(lower="kde_2d", diagonal="hist_1d", upper="scatter_2d"))
    ns.plot_2d(axac, label="posterior", color=post_color,
               kinds=dict(lower="kde_2d", diagonal="hist_1d", upper="scatter_2d"))
    # ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    if not single:
        ax[2].set(xlabel="$n$", ylabel=r"$\log{Z_n}$")

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
            ax[2].set(ylabel=r"$\log Z_n-\log Z_{\Lambda\text{CDM}}$")
            logR = logsumexp(pclogZs) - np.log(n)
            partials = np.e**(pclogZs - logsumexp(pclogZs))
            logRerr = (np.sum((partials * pclogZs)**2) + lcdm.logZerr**2)**(0.5)
            ax[2].set(title=f"$\\log Z_\\mathrm{{flexknot}} - \\log Z_{{\\Lambda\\text{{CDM}}}} = {logR:.2f} \\pm {logRerr:.2f}$")

            ax[2].set_xticks(idx[::5])
            ax[2].set_xticks(idx, minor=True)
        except FileNotFoundError:
            print("LCDM file not found :(")
        ax[2].errorbar(idx, pclogZs, yerr=pclogZerrs,
                       label='polychord',
                       marker='+', linestyle='None',
                       color=post_color)

    def fz(z, theta):
        theta = theta[~np.isnan(theta)]
        return fk(1/(1+z), theta)

    z = np.logspace(-3, np.log10(2.5))
    plot_lines(fz, z, prior[params], weights=prior.get_weights(),
               ax=ax[3], color=prior_color)
    plot_lines(fz, z, ns[params], weights=ns.get_weights(),
               ax=ax[3], color=post_color)
    ax[3].set(xlabel="$z$", ylabel="$w(z)$",
              xlim=(min(z), max(z)), ylim=(-3, 0),
              xscale='log')

    fig.suptitle(f"{'cobaya_' if cobaya else ''}{name}_{n}{'_i' if single else ''}")
    fig.tight_layout()
    plotpath = Path("plots/cobaya") / name if cobaya else Path("plots") / name
    plotpath.mkdir(parents=True, exist_ok=True)
    fig.savefig(plotpath / f"{name}_{n}{'_i' if single else ''}_wa.pdf",
                bbox_inches='tight')
    plt.show()
