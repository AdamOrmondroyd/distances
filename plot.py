import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import smplotlib
from fgivenx import plot_lines, plot_contours
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted
from pypolychord.output import PolyChordOutput
from common import flexknotparamnames
from flexknot import FlexKnot

if __name__ == "__main__":
    name = sys.argv[1]
    n = int(sys.argv[2])
    try:
        single = 'i' == sys.argv[3]
    except IndexError:
        single = False
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
    nss = [read_chains(f"chains/{name}_{i}") for i in idx]
    pcs = [PolyChordOutput("chains", f"{name}_{i}") for i in idx]
    prior = merge_samples_weighted([_ns.prior() for _ns in nss])
    # ns = merge_samples_weighted(nss)
    ns = merge_samples_weighted(nss, weights=[pc.logZ for pc in pcs])

if __name__ == "__main__":
    prior_color = 'C0'
    post_color = 'C1'
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax = ax.flatten()

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
    col = 'H0' if 'H0' in ns else 'H0rd'
    prior[col].plot.hist_1d(bins=40, ax=ax[1], alpha=0.5,
                            label='prior', color=prior_color)
    ns[col].plot.hist_1d(bins=40, ax=ax[1], alpha=0.5,
                         label='posterior', color=post_color)
    ax[1].set(xlabel=ns.get_label(col),
              xlim=(20, 100) if col == 'H0' else (3650, 18250), ylim=(0, 1.35))
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    if not single:
        logZs = []
        logZerrs = []
        for nsi in nss:
            logZi = nsi.logZ(nsamples=1_000)
            logZs.append(logZi.mean())
            logZerrs.append(logZi.std())
        ax[2].errorbar(idx, logZs, yerr=logZerrs,
                       label='anesthetic',
                       marker="+", linestyle="None")
        ax[2].set(xlabel="$N$", ylabel=r"$\log{Z_N}$")

        pclogZs = []
        pclogZerrs = []
        for i, pc in zip(idx, pcs):
            pclogZs.append(pc.logZ)
            pclogZerrs.append(pc.logZerr)
        ax[2].errorbar(idx, pclogZs, yerr=pclogZerrs,
                       label='polychord',
                       marker='+', linestyle='None',
                       color='m')
        ax[2].legend()

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

    fig.suptitle(f"{name}_{n}{'_i' if single else ''}")
    fig.tight_layout()
    plotpath = Path("plots") / name
    plotpath.mkdir(parents=True, exist_ok=True)
    fig.savefig(plotpath / f"{name}_{n}{'_i' if single else ''}_wa.pdf",
                bbox_inches='tight')
    plt.show()
