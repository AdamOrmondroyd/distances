import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted
from fgivenx import plot_lines
from distances import dl


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

omegar = 8.24e-5

params = [
    "Mb",
    "H0",
    "Omegam",
]

if n >= 2:
    params += ["wn"]

for i in range(n-2, 0, -1):
    params += [
        f"a{i}",
        f"w{i}",
    ]
if n >= 1:
    params += ["w0"]

print(ns)
print(f"{ns.columns=}")
print(f"{params=}")


def mu(z, theta):
    Mb, h0, omegam, *theta = theta
    theta = np.array(theta)
    theta = theta[~np.isnan(theta)]
    mu = 5 * np.log10(dl(z, h0, omegam, omegar, theta)) + 25
    return mu + Mb


fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]},
                        figsize=(8, 6))
z = np.logspace(-3, np.log10(2.5))
mu_lcdm = mu(z, [-19.3, 73.7, 0.33, -1])
ax = axs[0]
plot_lines(
    mu, z, ns[params],
    weights=ns.get_weights(),
    ax=ax,
    cache=f"cache/{name}_{n}_mb{'_i' if single else ''}",
    parallel=True,
)

df = pd.read_table('../clik_installs/desi/data/sn_data/PantheonPlus/Pantheon+SH0ES.dat', sep=' ', engine='python')
df = df[(df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)]

dfc = df[df['IS_CALIBRATOR'] == 0]
ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
        marker='+', markersize=0.5)

dfc = df[df['IS_CALIBRATOR'] == 1]
ax.plot(dfc['zHD'], dfc['m_b_corr'], linestyle="None",
        marker='+', markersize=0.5)

ax.set(xscale='log', xlabel=r"$z$", ylabel=r"$m_b$")

ax = axs[1]
plot_lines(
    lambda z, theta: mu(z, theta) - mu_lcdm, z, ns[params],
    weights=ns.get_weights(),
    ax=ax,
    cache=f"cache/{name}_{n}_∆mb{'_i' if single else ''}",
    parallel=True,
)

dfc = df[df['IS_CALIBRATOR'] == 0]
mu_lcdm = mu(dfc['zHD'].to_numpy(), [-19.3, 73.7, 0.33, -1])
ax.plot(dfc['zHD'], dfc['m_b_corr'] - mu_lcdm, linestyle="None",
        marker='+', markersize=0.5)

dfc = df[df['IS_CALIBRATOR'] == 1]
mu_lcdm = mu(dfc['zHD'].to_numpy(), [-19.3, 73.7, 0.33, -1])
ax.plot(dfc['zHD'], dfc['m_b_corr'] - mu_lcdm, linestyle="None",
        marker='+', markersize=0.5)

ax.set(xscale='log', xlabel=r"$z$", ylabel=r"$m_b - m_b(\Lambda\mathrm{CDM})$")
fig.tight_layout()
fig.savefig(f"plots/{name}_{n}_mb{'_i' if single else ''}.pdf", bbox_inches='tight')
plt.show()
