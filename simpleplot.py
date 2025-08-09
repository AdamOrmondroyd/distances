import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from common import flexknotparamnames
from fgivenx import plot_lines
from flexknot import FlexKnot
from plot import collect_chains, colors, plot_samples_dkl
import matplotlib.colors as mcolors
from matplotlib import gridspec


fk = FlexKnot(0, 1)


def fkplot(name, n, single, cobaya, ax, color='C1', label=None, useblack=False):
    idx, ns, nss, pcs, prior = collect_chains(name, n, single, cobaya, dodgy_wcdm=False, zenodo=True)
    np.random.seed(60017)
    ns = ns.compress(1000)
    prior = prior.compress(1000)

    ax[0].axhline(-1, color='k', linestyle='--')

    def f(x, theta):
        theta = theta[~np.isnan(theta)]
        return fk(x, theta)
    params = flexknotparamnames(n, tex=False)

    a = np.linspace(0, 1, 101)

    plot_samples_dkl(f, a, ns[params], prior[params], ax, color, max_alpha=0.5,)
                     # cache=f"{name}_simple")

    ax[0].set(xlim=(0, 1), ylim=(-3, 0), ylabel=r'$w(a)$')
    ax[1].set(xlabel=r'$a$', ylabel=r'$\mathcal{D}_\mathrm{KL}(\mathcal{P} || \pi)$')

    return ax


if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2)
    print(f"{gs=}")
    ax = []
    for _gs in gs:
        print(f"{_gs}")
        subgs = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=_gs,
            hspace=0.1,
            height_ratios=[2, 1],
        )
        ax0 = fig.add_subplot(subgs[0])
        ax1 = fig.add_subplot(subgs[1], sharex=ax0,
                              sharey=ax[0][1] if len(ax) else None,
                              )
        ax.append([ax0, ax1])
        plt.setp(ax[-1][0].get_xticklabels(), visible=False)

    n = int(sys.argv[1])

    for _ax, name, title in zip(
        ax,
        ["desiia_h0", "desi3ia_h0", "desides5y", "desi3des5y"],
        ["DESI DR1 + Pantheon+", "DESI DR2 + Pantheon+", "DESI DR1 + DES5Y", "DESI DR2 + DES5Y"],
    ):
        fkplot(name, n, False, False, _ax, colors[name])
        _ax[0].set(title=title)
    ax[0][1].set_ylim(0, 1.6)
    fig.tight_layout()
    plotpath = Path("plots") / name
    plotpath.mkdir(parents=True, exist_ok=True)
    fig.savefig('plots/all_together.pdf',
                bbox_inches='tight')
    plt.show()
