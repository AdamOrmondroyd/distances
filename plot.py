import sys
import numpy as np
from matplotlib import pyplot as plt
from fgivenx import plot_lines, plot_contours
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted
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

paramnames = []

if n >= 2:
    paramnames += [("wn", "w_n")]

for i in range(n-2, 0, -1):
    paramnames += [
        (f"a{i}", f"a_{i}"),
        (f"w{i}", f"w_{i}"),
    ]
if n >= 1:
    paramnames += [("w0", "w_0")]

params = [p[0] for p in paramnames]

if single:
    ns = read_chains(f"chains/{name}_{n}")
    ns = ns.prior()
else:
    idx = range(1, n+1)
    nss = [read_chains(f"chains/{name}_{i}") for i in idx]
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
    plot_lines(f, x, ns[params], weights=ns.get_weights(), ax=ax[0])
    # plot_contours(f, x, ns[params], weights=ns.get_weights(), ax=ax[1])
    for axi in ax[0], :  # ax[1]:
        axi.set(xlabel="$a$", ylabel="$w(a)$",
                xlim=(0, 1), ylim=(-3, 0))
    ns.H0.plot.hist_1d(bins=40, ax=ax[1])
    ax[1].set(xlabel='$H_0$')
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
    plot_lines(fz, z, ns[params], weights=ns.get_weights(), ax=ax[3])
    ax[3].set(xlabel="$z$", ylabel="$w(z)$",
              xlim=(min(z), max(z)), ylim=(-3, 0),
              xscale='log')

    fig.savefig(f"plots/{name}_{n}{'_i' if single else ''}_wa.pdf",
                bbox_inches='tight')
    plt.show()
