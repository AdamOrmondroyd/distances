import numpy as np
from scipy.stats import multivariate_normal
from desi import data, cov


zs = data.iloc[:, 0].to_numpy()
mean = data.iloc[:, 1].to_numpy()


samples = multivariate_normal(mean, cov).rvs(10_000)

dmidx = data.iloc[:, 2] == "DM_over_rs"
dhidx = data.iloc[:, 2] == "DH_over_rs"
dvidx = data.iloc[:, 2] == "DV_over_rs"
dm = samples[..., dmidx]
dh = samples[..., dhidx]
dv = samples[..., dvidx]
zmh = zs[dmidx]
zv = np.concatenate((zs[dvidx], zs[dhidx]))

zdm2dh13 = (zs[dmidx] * dm**2 * dh)**(1/3)
dv = np.hstack((dv, zdm2dh13))
sortidx = np.argsort(zv)
zv = zv[sortidx]
dv = dv[..., sortidx]

labelv = np.array(["BGS", "LRG", "LRG", "LRG+ELG",
                   "ELG", "QSO", r"Ly-$\alpha$"])
colorv = np.array([f"C{i}" for i in range(len(labelv))])

labelmh = labelv[[zvi in zmh for zvi in zv]]
colormh = colorv[[zvi in zmh for zvi in zv]]


def dvplot(ax):
    d0 = (dv / zv**(2/3))
    d0err = d0.std(axis=0)
    d0 = d0.mean(axis=0)
    for zi, d, derr, label, color in zip(zv, d0, d0err, labelv, colorv):
        ax.errorbar(zi, d, yerr=derr, marker="+", color=color, label=label)
    ax.set(xlabel=r"$z$", ylabel=r"$D_\text{V}/(r_\text{d}z^{2/3})$")
    return ax


def dmdhplot(ax):
    d1 = (dm / (dh * zmh))
    d1err = d1.std(axis=0)
    d1 = d1.mean(axis=0)
    for zi, d, derr, label, color in zip(zmh, d1, d1err, labelmh, colormh):
        ax.errorbar(zi, d, yerr=derr, marker="+", color=color, label=label)
    ax.set(xlabel=r"$z$", ylabel=r"$D_\text{M}/(zD_{H})$")
    return ax


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    dvplot(ax0)
    dmdhplot(ax1)
    ax0.legend()
    plt.show()
