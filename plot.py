import sys
import numpy as np
from matplotlib import pyplot as plt
import smplotlib
from fgivenx import plot_lines, plot_contours
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted
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
    prior = merge_samples_weighted([_ns.prior() for _ns in nss])
    ns = merge_samples_weighted(nss)

if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax = ax.flatten()

    fk = FlexKnot(0, 1)

    def f(x, theta):
        theta = theta[~np.isnan(theta)]
        return fk(x, theta)

    x = np.linspace(0, 1, 100)
    print(f"{ns[params]=}")
    plot_lines(f, x, prior[params], weights=prior.get_weights(),
               ax=ax[0], color='C1')
    plot_lines(f, x, ns[params], weights=ns.get_weights(),
               ax=ax[0], color='C0')
    # plot_contours(f, x, ns[params], weights=ns.get_weights(), ax=ax[1])
    for axi in ax[0], :  # ax[1]:
        axi.set(xlabel="$a$", ylabel="$w(a)$",
                xlim=(0, 1), ylim=(-3, 0))
    prior.H0.plot.hist_1d(bins=40, ax=ax[1], alpha=0.5,
                          label='prior', color='C1')
    ns.H0.plot.hist_1d(bins=40, ax=ax[1], alpha=0.5,
                       label='posterior', color='C0')
    ax[1].set(xlabel='$H_0$', xlim=(20, 100), ylim=(0, 1.35))
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    if not single:
        logZs = []
        logZerrs = []
        for nsi in nss:
            logZi = nsi.logZ(nsamples=1_000)
            logZs.append(logZi.mean())
            logZerrs.append(logZi.std())
        ax[2].errorbar(idx, logZs, yerr=logZerrs,
                       marker="+", linestyle="None")
        ax[2].set(xlabel="$N$", ylabel=r"$\log{Z_N}$")
        from pypolychord.output import PolyChordOutput

        pclogZs = []
        pclogZerrs = []
        for i in idx:
            pc = PolyChordOutput("chains", f"{name}_{i}")
            pclogZs.append(pc.logZ)
            pclogZerrs.append(pc.logZerr)
        ax[2].errorbar(idx, pclogZs, yerr=pclogZerrs,
                       marker='+', linestyle='None',
                       color='m')

    def fz(z, theta):
        theta = theta[~np.isnan(theta)]
        return fk(1/(1+z), theta)

    z = np.logspace(-3, np.log10(2.5))
    plot_lines(fz, z, prior[params], weights=prior.get_weights(),
               ax=ax[3], color='C1')
    plot_lines(fz, z, ns[params], weights=ns.get_weights(),
               ax=ax[3], color='C0')
    ax[3].set(xlabel="$z$", ylabel="$w(z)$",
              xlim=(min(z), max(z)), ylim=(-3, 0),
              xscale='log')

    fig.suptitle(f"{name}_{n}{'_i' if single else ''}")
    fig.tight_layout()
    fig.savefig(f"plots/{name}_{n}{'_i' if single else ''}_wa.pdf",
                bbox_inches='tight')
    plt.show()
