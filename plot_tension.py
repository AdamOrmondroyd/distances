import sys
import numpy as np
from matplotlib import pyplot as plt
import smplotlib
from anesthetic import read_chains

likelihoods = sys.argv[2:]

try:
    adaptive = True
    char = ""
    n = int(sys.argv[1])
except ValueError:
    adaptive = False
    char = sys.argv[1][0]
    n = int(sys.argv[1][1:])


lcdm_h0 = read_chains(f"chains/{''.join(likelihoods)}_lcdm")
lcdm_h1s = [read_chains(f"chains/{likelihood}_lcdm") for likelihood in likelihoods]

logZH0 = lcdm_h0.logZ(nsamples=1000)
logZH1 = np.sum(
    [lcdm_h1.logZ(nsamples=1000) for lcdm_h1 in lcdm_h1s],
    axis=0
)
logR = logZH0 - logZH1
logR = np.concatenate([logR.to_numpy()[None, ...],
                       np.load(f"{''.join(likelihoods)}_logR.npy")])

logLPH0 = lcdm_h0.logL_P(nsamples=1000)
logLPH1 = np.sum(
    [lcdm_h1.logL_P(nsamples=1000) for lcdm_h1 in lcdm_h1s],
    axis=0
)
logZ = np.concatenate([logZH0.to_numpy()[None, ...],
                       np.load(f"{''.join(likelihoods)}_logZ.npy")])
logZH1 = np.concatenate([logZH1[None, ...],
                         np.load(f"{''.join(likelihoods)}_logZH1.npy")])
logS = logLPH0 - logLPH1
logS = np.concatenate([logS.to_numpy()[None, ...],
                       np.load(f"{''.join(likelihoods)}_logS.npy")])

fig, ax = plt.subplots(3, figsize=(7, 10))

ax[0].errorbar(range(len(logZ)), logZ.mean(axis=-1),
               yerr=logZ.std(axis=-1),
               linestyle='None', marker='_',
               label=r"$H_0$")
ax[0].errorbar(range(len(logZH1)), logZH1.mean(axis=-1),
               yerr=logZH1.std(axis=-1),
               linestyle='None', marker='_',
               label=r"$H_1$")
ax[0].set(ylabel=r'$\log Z$')
ax[0].legend()

ax[1].errorbar(range(len(logR)), logR.mean(axis=-1),
               yerr=logR.std(axis=-1),
               linestyle='None', marker='_')
ax[1].set(ylabel=r'$\log R$')

ax[2].errorbar(range(len(logS)), logS.mean(axis=-1),
               yerr=logS.std(axis=-1),
               linestyle='None', marker='_')
ax[2].set(xlabel="$N$", ylabel=r'$\log S$')
idx = np.arange(len(logZ))
for axi in ax:
    axi.set_xticks(idx[::5])
    axi.set_xticklabels([r"$\Lambda$CDM"] + list(idx[::5][1:]))
    axi.set_xticks(idx, minor=True)

fig.tight_layout()
fig.savefig(f"plots/{''.join(likelihoods)}/{''.join(likelihoods)}_tension.pdf",
            bbox_inches='tight')
plt.show()
