import sys
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted
from fgivenx import plot_lines
from common import flexknotparamnames
from distances import dl
from plot import collect_chains
from ia import df, mb, zhd, zhel, invcov, omegar
from tqdm import tqdm
from joblib import Parallel, delayed
from flexknot import FlexKnot
import smplotlib
from pypolychord.output import PolyChordOutput
from scipy.stats import binned_statistic


color = '#d05a5c'
name = sys.argv[1]
try:
    n = int(sys.argv[2])
except ValueError:
    n = 'lcdm'
try:
    single = 'i' == sys.argv[3]
except IndexError:
    single = False

lcdm = read_chains(f"chains/{name}_lcdm")
idx, ns, nss, pcs, prior = collect_chains(name, n, single)

np.random.seed(1729)
ns = ns.compress()

# H0rd, Omegam, flexknot

lcdm_params = [
    "H0",
    "Omegam",
]

params = lcdm_params + flexknotparamnames(n, tex=False)

print(ns)
# ns = ns.iloc[np.argsort(ns.get_weights())]
print(f"{ns.columns=}")
print(f"{params=}")

# TODO: do calculations within fgivenx


def mu(zhd, zhel, theta):
    h0, omegam, *theta = theta
    theta = np.array(theta)
    theta = theta[~np.isnan(theta)]
    mu = 5 * np.log10(dl(zhd, zhel, h0, omegam, omegar, theta)) + 25
    return mu


rng = np.random.default_rng()


def mb_samples(ns, zhd, zhel, mb, lcdm=False, parallel=True):
    one = np.ones(len(invcov))[:, None]
    a = one.T @ invcov @ one
    a = a.squeeze()

    params_subset = params[:2] if lcdm else params
    params_arr = ns[params_subset].to_numpy()
    if parallel:
        mus = Parallel(n_jobs=-1, prefer="threads")(
            delayed(mu)(zhd, zhel, p) for p in tqdm(params_arr)
        )
    else:
        print("hello")
        mus = [mu(zhd, zhel, p) for p in tqdm(params_arr)]
        print("there")
    print(mus)
    mus = np.array(mus)
    print(f"{mus.shape=}")
    x = (mb - mus)
    xT = x[..., None, :]
    x = x[..., :, None]
    print(f"{x.shape=}")

    b = one.T @ invcov @ x + xT @ invcov @ one
    print(f"{a.shape=}")
    print(f"{b.squeeze().shape=}")
    b = b.squeeze()
    absolute_m = rng.normal(b/(2*a), 1/np.sqrt(a), len(b))  # , (73, len(b)))
    ns['absolute_m'] = absolute_m
    ns.set_label('absolute_m', r"$M_\mathrm{B}$")
    print(f"{absolute_m.shape=}")
    print(f"{absolute_m=}")

    return mus + absolute_m[..., None]


try:
    print("loading mb_samples")
    mb_ns = np.load(f"cache/{name}_{n}_mb_samples{'_i' if single else ''}.npy")
    ns['absolute_m'] = np.load(f"cache/{name}_{n}_absolute_m{'_i' if single else ''}.npy")
    ns.set_label('absolute_m', r"$M_\mathrm{B}$")
except FileNotFoundError:
    print("mb_samples not found. Regenerating:")
    mb_ns = mb_samples(ns, zhd, zhel, mb)
    np.save(f"cache/{name}_{n}_mb_samples{'_i' if single else ''}.npy", mb_ns)
    np.save(f"cache/{name}_{n}_absolute_m{'_i' if single else ''}.npy", ns.absolute_m.to_numpy())
    print("mb_samples generated")
print(f"{lcdm[-1:].shape=}")
mb_lcdm = mb_samples(lcdm[-3:], zhd, zhel, mb, lcdm=True, parallel=False)[-1]
print(f"{mb_lcdm=}")


fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharex=True)


df = df[(df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)]

ax = axs[0]

ahd = 1/(1+zhd)

fsamps = plot_lines(
    lambda a, mb: mb[ahd == a] - mb_lcdm[ahd == a],
    ahd,
    mb_ns,
    weights=ns.get_weights(),
    ax=ax,
    cache=f"cache/{name}_{n}_∆mb{'_i' if single else ''}",
    parallel=True,
    color=color,
)
mean = np.mean(fsamps, axis=-1)
sigma = np.std(fsamps, axis=-1)
ax.plot(ahd, mean, color=color, linestyle='--')
ax.fill_between(ahd, mean-sigma, mean+sigma, color=color, alpha=0.5)

dfc = df[df['IS_CALIBRATOR'] == 0]
ax.plot(1/(1+dfc['zHD']), dfc['m_b_corr']-mb_lcdm, linestyle="None",
        marker='+', markersize=0.5, color='C0')
binned_mb, bin_edges, _ = binned_statistic(1/(1+dfc['zHD']), dfc['m_b_corr']-mb_lcdm, bins=20)
binned_a = (bin_edges[1:] + bin_edges[:-1]) / 2
ax.stairs(binned_mb, bin_edges, color='k', label=r'binned supernovae')
ax.axhline(0, color="C0", linestyle="--")
ax.set(xlabel=r"$a$", ylabel=r"$m_B-{m_B}_{\Lambda\mathrm{CDM}}$",
       # xlim=(np.min(zhd), np.max(zhd)), ylim=(-0.1, 0.1))
       xlim=(np.min(ahd), np.max(ahd)), ylim=(-0.1, 0.1))
ax.legend(fontsize='small', loc='upper right', frameon=True, framealpha=0.75)

# dfc = df[df['IS_CALIBRATOR'] == 0]
# ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
        # marker='+', markersize=0.5)

# dfc = df[df['IS_CALIBRATOR'] == 1]
# ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
        # marker='+', markersize=0.5)


ax = axs[1]
print(f"{ns.absolute_m}")
# ns.absolute_m.plot.hist_1d(ax=ax, bins=50)
# ax.set(xlabel=r"$M_\mathrm{B}$")

fk = FlexKnot(0, 1)


def wz(a, theta):
    theta = theta[~np.isnan(theta)]
    return fk(a, theta)


fsamps = plot_lines(wz, ahd, ns[params[2:]].to_numpy(), weights=ns.get_weights(), ax=ax, color=color)
mean = np.mean(fsamps, axis=-1)
sigma = np.std(fsamps, axis=-1)
ax.plot(ahd, mean, color=color, linestyle='--')
ax.fill_between(ahd, mean-sigma, mean+sigma, color=color, alpha=0.5)
ax.axhline(-1, color="C0", linestyle="--")
ax.set(ylim=(-3, 0), ylabel=r"$w(a)$", xlabel=r"$a$")
fig.tight_layout()
fig.savefig(f"plots/{name}/{name}_{n}_mb{'_i' if single else ''}.pdf", bbox_inches='tight')
plt.show()
