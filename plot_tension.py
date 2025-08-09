import sys
import numpy as np
from matplotlib import pyplot as plt
import smplotlib
from anesthetic import read_chains
from pathlib import Path


def tensionplot(likelihoods, n, ax, color, label):

    lcdm_h0 = read_chains(f"chains/{'_'.join(likelihoods)}/{'_'.join(likelihoods)}_lcdm")
    lcdm_h1s = [read_chains(f"chains/{likelihood}/{likelihood}_lcdm") for likelihood in likelihoods]

    logZH0 = lcdm_h0.logZ(nsamples=1000)
    logZH1 = np.sum(
        [lcdm_h1.logZ(nsamples=1000) for lcdm_h1 in lcdm_h1s],
        axis=0
    )
    logR = logZH0 - logZH1
    logRi = np.concatenate([logR.to_numpy()[None, ...],
                            np.load(f"{'_'.join(likelihoods)}_logRi.npy")])
    logR = np.concatenate([logR.to_numpy()[None, ...],
                           np.load(f"{'_'.join(likelihoods)}_logR.npy")])

    logLPH0 = lcdm_h0.logL_P(nsamples=1000)
    logLPH1 = np.sum(
        [lcdm_h1.logL_P(nsamples=1000) for lcdm_h1 in lcdm_h1s],
        axis=0
    )
    logZi = np.concatenate([logZH0.to_numpy()[None, ...],
                            np.load(f"{'_'.join(likelihoods)}_logZi.npy")])
    logZi -= logZH0.to_numpy()[None, :]
    logZ = np.concatenate([logZH0.to_numpy()[None, ...],
                           np.load(f"{'_'.join(likelihoods)}_logZ.npy")])
    logZ -= logZH0.to_numpy()[None, :]
    logZH1i = np.concatenate([logZH1[None, ...],
                              np.load(f"{'_'.join(likelihoods)}_logZH1i.npy")])
    logZH1i -= logZH0.to_numpy()[None, :]
    logZH1 = np.concatenate([logZH1[None, ...],
                             np.load(f"{'_'.join(likelihoods)}_logZH1.npy")])
    logZH1 -= logZH0.to_numpy()[None, :]

    logS = logLPH0 - logLPH1
    logSi = np.concatenate([logS.to_numpy()[None, ...],
                            np.load(f"{'_'.join(likelihoods)}_logSi.npy")])
    logS = np.concatenate([logS.to_numpy()[None, ...],
                           np.load(f"{'_'.join(likelihoods)}_logS.npy")])


    ax.errorbar(np.arange(len(logRi[1:]))+1, logRi[1:].mean(axis=-1),
                yerr=logRi[1:].std(axis=-1),
                linestyle='None', marker='_',
                color=color, label=label)

    ax.set(xlabel="$n$", ylabel=r'$\log R$')
    idx = np.arange(len(logZ[1:]))+1
    print(idx)
    ax.set_xticks(idx[4::5])
    # ax.set_xticklabels([r"$\Lambda$CDM"] + list(idx[::5][1:]))
    # ax.set_xticklabels(list(idx[::5][1:]))
    ax.set_xticks(idx, minor=True)
    logRlcdm = logR[0]
    ax.axhline(logRlcdm.mean(), color=color, linestyle='--')
    ax.text(n-2, logRlcdm.mean()+0.1, r'$\Lambda$CDM', color=color)



if __name__ == "__main__":
    try:
        adaptive = True
        char = ""
        n = int(sys.argv[1])
    except ValueError:
        adaptive = False
        char = sys.argv[1][0]
        n = int(sys.argv[1][1:])
    likelihoods = sys.argv[2:]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    desi_color = '#58acbc'
    garter_blue = '#1f77b4'  # for desi+ia
    des5y_color = 'C1'
    newiacolor = '#ff964f'
    purple = '#7B0043'
    newdescolor = '#caa0ff'
    tensionplot(["desidr1", "pantheonplus"], n, ax[0], garter_blue, 'DESI DR1 vs Pantheon+')
    tensionplot(["desidr2", "pantheonplus"], n, ax[0], newiacolor, "DESI DR2 vs Pantheon+")
    tensionplot(["desidr1", "des5y"], n, ax[1], purple, 'DESI DR1 vs DES5Y')
    tensionplot(["desidr2", "des5y"], n, ax[1], newdescolor, 'DESI DR2 vs DES5Y')
    ax[0].set(title="DESI vs Pantheon+")
    ax[1].set(title="DESI vs DES5Y")
    for _ax in ax:
        _ax.legend(frameon=True, framealpha=0.5)
    ax[1].tick_params(labelleft=True)
    fig.tight_layout()

    (Path("plots") / '_'.join(likelihoods)).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"plots/{'_'.join(likelihoods)}/{'_'.join(likelihoods)}_tension_i.pdf",
                bbox_inches='tight')
    plt.show()
