import sys
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted
from fgivenx import plot_lines
from common import flexknotparamnames
from distances import dl
from ia import df, mb, zhd, zhel, invcov, omegar
from tqdm import tqdm
from joblib import Parallel, delayed
from flexknot import FlexKnot
import smplotlib
from pypolychord.output import PolyChordOutput


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
if single:
    idx = [n]
    ns = read_chains(f"chains/{name}_{n}")
else:
    idx = range(1, n+1)
    nss = [read_chains(f"chains/{name}_{n}") for i in idx]
    pcs = [PolyChordOutput("chains", f"{name}_{i}") for i in idx]
    ns = merge_samples_weighted(nss, weights=[pc.logZ for pc in pcs])
    ns = merge_samples_weighted(nss)

ns = ns.compress()

# H0rd, Omegam, flexknot

lcdm_params = [
    "H0",
    "Omegam",
]

params = lcdm_params + flexknotparamnames(n, tex=False)

print(ns)
ns = ns.iloc[np.argsort(ns.get_weights())]
# ns = ns.iloc[-100:]
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


fig, axs = plt.subplots(3, gridspec_kw={'height_ratios': [3, 1, 3]},
                        figsize=(8, 10))


ax = axs[0]
plot_lines(
    lambda z, mb: mb[zhd == z],
    zhd,
    mb_ns,
    weights=ns.get_weights(),
    ax=ax,
    cache=f"cache/{name}_{n}_mb{'_i' if single else ''}",
    parallel=True,
)
ax.plot(zhd, mb_lcdm)

df = df[(df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)]
#
dfc = df[df['IS_CALIBRATOR'] == 0]
ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
        marker='+', markersize=0.5)

# dfc = df[df['IS_CALIBRATOR'] == 1]
# ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
#         marker='+', markersize=0.5)

ax.set(xscale='log', xlabel=r"$z$", ylabel=r"$m_b$")

ax = axs[1]


plot_lines(
    lambda z, mb: mb[zhd == z] - mb_lcdm[zhd == z],
    zhd,
    mb_ns,
    weights=ns.get_weights(),
    ax=ax,
    cache=f"cache/{name}_{n}_∆mb{'_i' if single else ''}",
    parallel=True,
)

dfc = df[df['IS_CALIBRATOR'] == 0]
ax.plot(dfc['zHD'], dfc['m_b_corr']-mb_lcdm, linestyle="None",
        marker='+', markersize=0.5)
ax.set(xscale='log', xlabel=r"$z$", ylabel=r"$m_B-{m_B}_{\Lambda\mathrm{CDM}}$")

# dfc = df[df['IS_CALIBRATOR'] == 0]
# ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
        # marker='+', markersize=0.5)

# dfc = df[df['IS_CALIBRATOR'] == 1]
# ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
        # marker='+', markersize=0.5)

ax.set(xscale='log', xlabel=r"$z$", ylabel=r"$m_b - m_b(\Lambda\mathrm{CDM})$")

ax = axs[2]
print(f"{ns.absolute_m}")
ns.absolute_m.plot.hist_1d(ax=ax, bins=50)
ax.set(xlabel=r"$M_\mathrm{B}$")
fig.tight_layout()
fig.savefig(f"plots/{name}/{name}_{n}_mb{'_i' if single else ''}.pdf", bbox_inches='tight')
plt.show()
