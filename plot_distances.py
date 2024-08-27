import sys
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted
from fgivenx import plot_lines
from distances import dh_over_rs, dm_over_rs, dv_over_rs
from bao import dmdhplot, dvplot


name = sys.argv[1]
n = int(sys.argv[2])
try:
    single = 'i' == sys.argv[3]
except IndexError:
    single = False

if single:
    ns = read_chains(f"chains/{name}_{n}")
else:
    idx = [1, 2, 5, 6, 7, 8, 9]
    nss = [read_chains(f"chains/{name}_{n}") for i in idx]
    ns = merge_samples_weighted(nss)

# H0rd, Omegam, flexknot

omegar = 8.24e-5

params = [
    "H0rd",
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


def dm_over_zdh(z, theta):
    h0rd, omegam, *theta = theta
    theta = np.array(theta)
    theta = theta[~np.isnan(theta)]

    return [
        dm_over_rs(zi, h0rd, omegam, omegar, theta)
        / (zi * dh_over_rs(zi, h0rd, omegam, omegar, theta))
        for zi in z
    ]


def dv_over_rdz23(z, theta):
    h0rd, omegam, *theta = theta
    theta = np.array(theta)
    theta = theta[~np.isnan(theta)]

    return [
        dv_over_rs(zi, h0rd, omegam, omegar, theta) / zi**(2/3) for zi in z
    ]


fig, axs = plt.subplots(1, 2, figsize=(12, 6))
z = np.linspace(0.01, 2.5)
plot_lines(dm_over_zdh, z, ns[params], weights=ns.get_weights(), ax=axs[0],
           cache=f"cache/{name}_dm_over_zdh_{n}{'_i' if single else ''}",
           parallel=True)
plot_lines(dv_over_rdz23, z, ns[params], weights=ns.get_weights(), ax=axs[1],
           cache=f"cache/{name}_dv_over_rdz23_{n}{'_i' if single else ''}",
           parallel=True)
for ax in axs:
    ax.set(xlim=(0, 2.5))
dmdhplot(axs[0])
dvplot(axs[1])
# axs[0].set(ylim=(17, 21.5))
# axs[1].set(ylim=(0.95, 2.1))


fig.tight_layout()
fig.savefig(f"plots/{name}_{n}_distances{'_i' if single else ''}.pdf",
            bbox_inches='tight')
plt.show()
