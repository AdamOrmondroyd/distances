from numpy import array, log, log10, ones, pi
from numpy.linalg import inv, slogdet
from distances import c, h0_dl_over_c


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

        one = ones(len(self.cov))[:, None]
        invcov = inv(self.cov)
        self.invcov_tilde = (
            invcov - invcov @ one @ one.T @ invcov / (one.T @ invcov @ one)
        )
        self.lognormalisation = 0.5 * (
            log(2*pi) - slogdet(2 * pi * self.cov)[1]
            - log((one.T @ invcov @ one).squeeze())
        ) + log(c / (1e-5 * (h0max - h0min)))

        self.a = 1e-5 * h0min / c
        self.b = 1e-5 * h0max / c
        self.onesigma_times_5_over_log10 = (
            one.T @ self.invcov_tilde * 5 / log(10)
        )

    def _y(self, omegam, omegar, theta=array([-1])):
        theta = array(theta)
        return 5 * log10(
            h0_dl_over_c(self.zhd, self.zhel, omegam, omegar, theta)) - self.mb

    def __call__(self, omegam, omegar, theta=array([-1])):
        y = self._y(omegam, omegar, theta)
        capital_y = float((self.onesigma_times_5_over_log10 @ y).squeeze())
        return (
            - float(y.T @ self.invcov_tilde @ y / 2)
            + log(
                    (self.b**(capital_y + 1) - self.a**(capital_y + 1))
                    / (capital_y + 1)
                )
            + self.lognormalisation)
