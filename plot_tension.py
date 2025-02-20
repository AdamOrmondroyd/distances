import sys
import numpy as np
from matplotlib import pyplot as plt
import smplotlib
from anesthetic import read_chains


def tensionplot(likelihoods, n, ax):

    lcdm_h0 = read_chains(f"chains/{''.join(likelihoods)}_lcdm")
    lcdm_h1s = [read_chains(f"chains/{likelihood}_lcdm") for likelihood in likelihoods]

    logZH0 = lcdm_h0.logZ(nsamples=1000)
    logZH1 = np.sum(
        [lcdm_h1.logZ(nsamples=1000) for lcdm_h1 in lcdm_h1s],
        axis=0
    )
    logR = logZH0 - logZH1
    logRi = np.concatenate([logR.to_numpy()[None, ...],
                            np.load(f"{''.join(likelihoods)}_logRi.npy")])
    logR = np.concatenate([logR.to_numpy()[None, ...],
                           np.load(f"{''.join(likelihoods)}_logR.npy")])

    logLPH0 = lcdm_h0.logL_P(nsamples=1000)
    logLPH1 = np.sum(
        [lcdm_h1.logL_P(nsamples=1000) for lcdm_h1 in lcdm_h1s],
        axis=0
    )
    logZi = np.concatenate([logZH0.to_numpy()[None, ...],
                            np.load(f"{''.join(likelihoods)}_logZi.npy")])
    logZi -= logZH0.to_numpy()[None, :]
    logZ = np.concatenate([logZH0.to_numpy()[None, ...],
                           np.load(f"{''.join(likelihoods)}_logZ.npy")])
    logZ -= logZH0.to_numpy()[None, :]
    logZH1i = np.concatenate([logZH1[None, ...],
                              np.load(f"{''.join(likelihoods)}_logZH1i.npy")])
    logZH1i -= logZH0.to_numpy()[None, :]
    logZH1 = np.concatenate([logZH1[None, ...],
                             np.load(f"{''.join(likelihoods)}_logZH1.npy")])
    logZH1 -= logZH0.to_numpy()[None, :]

    logS = logLPH0 - logLPH1
    logSi = np.concatenate([logS.to_numpy()[None, ...],
                            np.load(f"{''.join(likelihoods)}_logSi.npy")])
    logS = np.concatenate([logS.to_numpy()[None, ...],
                           np.load(f"{''.join(likelihoods)}_logS.npy")])

    ax[0].errorbar(np.arange(len(logZ[1:]))+1, logZ.mean(axis=-1)[1:],
                   yerr=logZ.std(axis=-1)[1:],
                   linestyle='None', marker='_',
                   color="C0",
                   label=r"$H_0$")
    ax[0].errorbar(np.arange(len(logZi[2:]))+2, logZi[2:].mean(axis=-1),
                   yerr=logZi[2:].std(axis=-1),
                   linestyle='None', marker='_',
                   color="C0",
                   alpha=0.5)
    ax[0].errorbar(range(len(logZH1)), logZH1.mean(axis=-1),
                   yerr=logZH1.std(axis=-1),
                   linestyle='None', marker='_',
                   color="C1",
                   label=r"$H_1$")
    ax[0].errorbar(np.arange(len(logZH1i[2:]))+2, logZH1i[2:].mean(axis=-1),
                   yerr=logZH1i[2:].std(axis=-1),
                   linestyle='None', marker='_',
                   color="C1",
                   alpha=0.5)
    ax[0].set(ylabel=r'$\log Z - \log Z(H_0)_{\Lambda\mathrm{CDM}}$')
    ax[0].legend()

    ax[1].errorbar(range(len(logR)), logR.mean(axis=-1),
                   yerr=logR.std(axis=-1),
                   linestyle='None', marker='_',
                   color='C0')
    ax[1].errorbar(np.arange(len(logRi[2:]))+2, logRi[2:].mean(axis=-1),
                   yerr=logRi[2:].std(axis=-1),
                   linestyle='None', marker='_',
                   color='C0', alpha=0.5)
    ax[1].set(ylabel=r'$\log R$')

    ax[2].errorbar(range(len(logS)), logS.mean(axis=-1),
                   yerr=logS.std(axis=-1),
                   linestyle='None', marker='_',
                   color='C0')
    ax[2].errorbar(np.arange(len(logSi[2:]))+2, logSi[2:].mean(axis=-1),
                   yerr=logSi[2:].std(axis=-1),
                   linestyle='None', marker='_',
                   color='C0', alpha=0.5)
    ax[2].set(xlabel="$N$", ylabel=r'$\log S$')
    idx = np.arange(len(logZ))
    for axi in ax:
        axi.set_xticks(idx[::5])
        axi.set_xticklabels([r"$\Lambda$CDM"] + list(idx[::5][1:]))
        axi.set_xticks(idx, minor=True)


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
    fig, ax = plt.subplots(3, 2, figsize=(9, 10))
    tensionplot([likelihoods[0], likelihoods[1]], n, ax[:, 0])
    tensionplot([likelihoods[0], likelihoods[2]], n, ax[:, 1])
    ax[0, 0].set(title="DESI vs Pantheon+")
    ax[0, 1].set(title="DESI vs DES5Y")
    fig.tight_layout()
    fig.savefig(f"plots/{''.join(likelihoods)}/{''.join(likelihoods)}_tension_i.pdf",
                bbox_inches='tight')
    plt.show()
