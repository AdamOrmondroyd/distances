from numpy import array, diag, log, log10, ones, pi
from numpy.linalg import inv, slogdet, solve
from scipy.linalg import ldl
from distances import h0_dl_over_c


class IaLogL:
    def __init__(self, df, cov, mb_column, z_cutoff=0.0,
                 h0min=20, h0max=100):

        self.df = df
        self.cov = cov

        mask = df['zHD'] > z_cutoff
        self.mb = df[mb_column].to_numpy()[mask]
        self.zhd = df['zHD'].to_numpy()[mask]
        self.zhel = df['zHEL'].to_numpy()[mask]

        self.cov = cov[mask, :][:, mask]

        (self.lT, self.d, self.perm), self.lognorm = self._compute_cholesky_and_lognorm(cov)

    def _compute_cholesky_and_lognorm(self, cov):
        one = ones((len(cov), 1))

        invcov = inv(cov)
        invcov_one = solve(cov, one)  # More stable than inv @ one
        one_T_invcov_one = invcov_one.sum(keepdims=True)

        # Constrained inverse using Cobaya's more stable approach
        # C^-1_tilde = C^-1 - (C^-1 @ 1) @ solve(1^T @ C^-1 @ 1, (C^-1 @ 1)^T)
        invcov_tilde = (
            invcov
            - invcov_one @ solve(one_T_invcov_one, invcov_one.T)
        )

        # Compute Cholesky decomposition for GPU vmap bug fix
        # This avoids the problematic y.T @ M @ y operation
        l, d, perm = ldl(invcov_tilde)
        lT = array(l).T
        d = array(diag(d))

        # Compute log normalization in fp64
        sign, logdet = slogdet(cov)
        if sign != 1:
            raise ValueError("Covariance matrix must be positive definite.")
        lognorm = -0.5 * (
            logdet                           # log|C|
            + log(2*pi) * (len(cov) - 1)  # log(2π)^(n-1) after marginalization
            + log(one_T_invcov_one.item())       # log(1^T C^-1 1)
        )
        return (lT, d, perm), lognorm

    def _y(self, omegam, omegar, theta=array([-1])):
        theta = array(theta)
        return 5 * log10(
            h0_dl_over_c(self.zhd, self.zhel, omegam, omegar, theta)) - self.mb

    def __call__(self, *args, **kwargs):
        y = self._y(*args, **kwargs)
        v = self.lT @ y[self.perm]
        return -(v * self.d * v).sum() / 2.0 + self.lognorm
