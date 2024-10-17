import sys
import numpy as np
from tqdm import tqdm
from desi import zs, di_over_rss, omegar
from ia import zhd, zhel
from distances import dl


# for Ia supernovae, I don't think I can marginalize my way around M_b
lower = np.array([3650, 20, 0.01, -20])
upper = np.array([18250, 100, 0.99, -18])
prior_range = upper - lower

nsamples = int(sys.argv[1])


def generic_prior(x):
    return lower + x * prior_range


rng = np.random.default_rng()
x = rng.uniform(size=(nsamples, len(prior_range)))
thetas = generic_prior(x)

desi_data = np.array([
    [di_over_rs(z, *theta, omegar, np.array([-1]))
        for z, di_over_rs in zip(zs, di_over_rss)]
    for theta in tqdm(thetas[..., [0, 2]])
])


ia_data = np.array([
    absolute_m + 5 * np.log10(
        dl(zhd, zhel, h0, omegam, omegar, np.array([-1]))) + 25
    for h0, omegam, absolute_m in tqdm(thetas[..., [1, 2, 3]])
])

data = np.hstack([desi_data, ia_data])
np.save("sbi_data.npy", data)
