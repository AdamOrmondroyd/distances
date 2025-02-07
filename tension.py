import sys
import numpy as np
from numpy import e
from scipy.special import logsumexp
from anesthetic import read_chains
from tqdm import tqdm


likelihoods = sys.argv[2:]

try:
    adaptive = True
    char = ""
    n = int(sys.argv[1])
except ValueError:
    adaptive = False
    char = sys.argv[1][0]
    n = int(sys.argv[1][1:])


def read_chainss(name, n):
    return [
        read_chains(f"chains/{name}_{i}")
        for i in tqdm(range(1, n+1))
    ]


def logZ(nss, n):
    return logsumexp(
        [nsi.logZ(nsamples=1000) for nsi in tqdm(nss)], axis=0
    ) - np.log(n)


def logLP(nss, n):
    logLPi = np.array([nsi.logL_P(nsamples=1000) for nsi in nss])
    logZi = np.array([nsi.logZ(nsamples=1000) for nsi in nss])
    logZi -= np.max(logZi)

    # pull out factor of most extreme logZ for logsumexp-style stability
    # since it factors out
    # maxlogZi = logZi[np.argmax(np.abs(logZi), axis=1)]
    # assert np.isclose(np.sum(logLPi * e**logZi) / np.sum(e**logZi),
    #                   np.sum(logLPi * e**(logZi - maxlogZi))
    #                   / np.sum(e**(logZi - maxlogZi)))
    return np.sum(logLPi * e**logZi, axis=0) / np.sum(e**logZi, axis=0)
    # return np.sum(logLPi * e**(logZi - maxlogZi), axis=0) / np.sum(e**(logZi - maxlogZi), axis=0)


logZs = []
logZH1s = []
logRs = []
logSs = []
nssH0 = read_chainss("".join(likelihoods), n)
nssH1s = [read_chainss(likelihood, n) for likelihood in likelihoods]

for i in range(1, n+1):
    print(f"{i=}")
    logZH0 = logZ(nssH0[i-1:i], i)
    logZH1 = np.sum(
        [logZ(nssH1[i-1:i], i) for nssH1 in nssH1s],
        axis=0,
    )
    logZs.append(logZH0)
    logZH1s.append(logZH1)
    print(f"{logZH0.mean()}±{logZH0.std()}")
    print(f"{logZH1.mean()}±{logZH1.std()}")
    logR = logZH0-logZH1
    print(f"logR={logR.mean()}±{logR.std()}")
    logRs.append(logR)

    logLPH0 = logLP(nssH0[i-1:i], i)
    logLPH1 = np.sum(
        [logLP(nssH1[:i], i) for nssH1 in nssH1s],
        axis=0,
    )
    print(f"{logLPH0.mean()}±{logLPH0.std()}")
    print(f"{logLPH1.mean()}±{logLPH1.std()}")
    logS = logLPH0-logLPH1
    print(f"logS={logS.mean()}±{logS.std()}")
    logSs.append(logS)

np.save(f"{''.join(likelihoods)}_logZ.npy", logZs)
np.save(f"{''.join(likelihoods)}_logZH1.npy", logZH1s)
np.save(f"{''.join(likelihoods)}_logR.npy", logRs)
np.save(f"{''.join(likelihoods)}_logS.npy", logSs)
