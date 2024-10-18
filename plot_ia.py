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


name = sys.argv[1]
n = int(sys.argv[2])
try:
    single = 'i' == sys.argv[3]
except IndexError:
    single = False

if single:
    ns = read_chains(f"chains/{name}_{n}")
else:
    nss = [read_chains(f"chains/{name}_{n}") for i in range(1, n+1)]
    ns = merge_samples_weighted(nss)

# H0rd, Omegam, flexknot

params = [
    "H0",
    "Omegam",
]

params += flexknotparamnames(n, False)

print(ns)
ns = ns.iloc[np.argsort(ns.get_weights())]
ns = ns.iloc[-500:]
print(f"{ns.columns=}")
print(f"{params=}")

one = np.ones(len(invcov))[:, None]
a = one.T @ invcov @ one
a = a.squeeze()

# TODO: do calculations within fgivenx


def mu(zhd, zhel, theta):
    h0, omegam, *theta = theta
    theta = np.array(theta)
    theta = theta[~np.isnan(theta)]
    mu = 5 * np.log10(dl(zhd, zhel, h0, omegam, omegar, theta)) + 25
    return mu


mus = np.array([mu(zhd, zhel, p) for p in tqdm(ns[params].to_numpy())])
print(f"{mus.shape=}")
x = (mb - mus)
xT = x[..., None, :]
x = x[..., :, None]
print(f"{x.shape=}")

b = one.T @ invcov @ x + xT @ invcov @ one
print(f"{a.shape=}")
print(f"{b.squeeze().shape=}")
b = b.squeeze()
rng = np.random.default_rng()
absolute_m = rng.normal(b/(2*a), 1/np.sqrt(a), len(b))  # , (73, len(b)))
print(f"{absolute_m.shape=}")
print(f"{absolute_m=}")

mb_samples = mus + absolute_m[..., None]
print(f"{mb_samples.shape=}")


fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]},
                        figsize=(8, 6))
m_lcdm = mu(zhd, zhel, [73.7, 0.33, -1]) + 19.3

ax = axs[0]
plot_lines(
    lambda z, mb: mb[zhd == z],
    zhd,
    mb_samples,
    weights=ns.get_weights(),
    ax=ax,
    cache=f"cache/{name}_{n}_mb{'_i' if single else ''}",
    parallel=True,
)

df = df[(df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)]
#
dfc = df[df['IS_CALIBRATOR'] == 0]
ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
        marker='+', markersize=0.5)

dfc = df[df['IS_CALIBRATOR'] == 1]
ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
        marker='+', markersize=0.5)

ax.set(xscale='log', xlabel=r"$z$", ylabel=r"$m_b$")

ax = axs[1]


plot_lines(
    lambda z, mb: mb[zhd == z] - m_lcdm[zhd == z],
    zhd,
    mb_samples,
    weights=ns.get_weights(),
    ax=ax,
    cache=f"cache/{name}_{n}_∆mb{'_i' if single else ''}",
    parallel=True,
)

dfc = df[df['IS_CALIBRATOR'] == 0]
m_lcdm = mu(dfc['zHD'].to_numpy(), dfc['zHEL'].to_numpy(),
            [73.7, 0.33, -1]) + 19.3
ax.plot(dfc['zHD'], dfc['m_b_corr'] - m_lcdm, linestyle="None",
        marker='+', markersize=0.5)

dfc = df[df['IS_CALIBRATOR'] == 1]
m_lcdm = mu(dfc['zHD'].to_numpy(), dfc['zHEL'].to_numpy(),
            [73.7, 0.33, -1]) + 19.3
ax.plot(dfc['zHD'], dfc['m_b_corr'] - m_lcdm, linestyle="None",
        marker='+', markersize=0.5)

ax.set(xscale='log', xlabel=r"$z$", ylabel=r"$m_b - m_b(\Lambda\mathrm{CDM})$")
fig.tight_layout()
fig.savefig(f"plots/{name}_{n}_mb{'_i' if single else ''}.pdf", bbox_inches='tight')
plt.show()
