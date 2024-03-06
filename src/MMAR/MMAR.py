from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import njit, prange
from numpy.random import normal
from scipy.optimize import minimize_scalar, root_scalar
from scipy.stats import jarque_bera, norm, zscore
from statsmodels.tsa.stattools import adfuller, kpss


class MMAR:
    def __init__(
        self, price: pd.Series, seed: int = 42, volume: pd.Series | None = None, silent:bool=False
    ) -> None:
        """Class to build a MMAR model starting from actual data

        Args:
            price (pd.Series): price series
            seed (int, optional): seed. Defaults to 42.
            volume (pd.Series | None, optional): volume series. Defaults to None.
            silent (bool, optional): wheter to print warnings. Defaults to False.

        Returns:
            None
        """
        self.price = price
        self._log_prices: np.ndarray[float, Any] = np.log(
            cast(np.ndarray, price.values)
        )
        self._theta: np.ndarray[float, Any] | None = None
        self._m: float | None = None
        self._sigma: float | None = None
        self._sigma_ret: float | None = None
        self._mu: float | None = None
        self._q: np.ndarray[float, Any] | None = None
        self._c: np.ndarray[float, Any] | None = None
        self._tau: np.ndarray[float, Any] | None = None
        self._alpha_min: float | None = None
        self.seed: int = seed
        if volume is not None:
            self._volume = volume.values
        else:
            self._volume = volume
        self._H: float | None = None
        self.silent = silent
        self.post_init()

    def post_init(self) -> None:
        """Post init function to init vars"""
        n = len(self.price)
        m = n
        while len(self.divisors(m)) < 5:
            m -= 1
        if m != n:
            if not self.silent:
                print(
                    f"The series has been adjusted: the orginal size was {n}, the new size is {m} with {len(self.divisors(m))} dividers."
                )
            self.price = self.price[-m:]
            self._log_prices = np.log(cast(np.ndarray, self.price.values))
        self.config()

    # Properties
    @property
    def theta(self) -> np.ndarray[float, Any]:
        """Theta values

        Returns:
            np.ndarray[float, Any]: Thetas
        """
        assert self._theta is not None
        return self._theta

    @property
    def H(self) -> float:
        """The Hurst exponent

        Returns:
            float: the Hurst exponent
        """
        assert self._H is not None
        return self._H

    @property
    def mu(self) -> float:
        """Mu of alpha

        .. math::

            \\mu_{\\alpha} = \\frac {m_{\\alpha}}{H}

            \\textrm{where } f_{\\theta}(\\mu_{\\alpha}) = 1

        .. note::
            It's also called :math:`\\lambda`
        Returns:
            float: Mu
        """
        assert self._mu is not None
        return self._mu

    @property
    def m(self) -> float:
        """M of alpha

        Is the first exponent that makes the multifractal spectrum = 1

        .. math::

            m_{\\alpha}

        .. note::
            It's also called :math:`\\alpha_{0}`

        Returns:
            float: m of alpha
        """
        assert self._m is not None
        return self._m

    @property
    def sigma(self) -> float:
        """Sigma of alpha

        .. math::

            \\sigma_{\\alpha} = \\sqrt {\\frac {2 (\\mu_{\\alpha}-1)} {\\log(b)}}

        Returns:
            float: Sigma of alpha
        """
        assert self._sigma is not None
        return self._sigma

    @property
    def sigma_ret(self) -> float:
        """Instant volatiltiy of returns

        Returns:
            float: volatility of returns
        """
        assert self._sigma_ret is not None
        return self._sigma_ret

    @property
    def alpha_min(self) -> float:
        """Alpha min

        The minimum allowable value for alpha

        Returns:
            float: alpha min
        """
        assert self._alpha_min is not None
        return self._alpha_min

    @property
    def tau(self) -> np.ndarray[float, Any]:
        """Tau values
        The value for the scaling function

        .. math::

            \\tau(q)

        Returns:
            np.ndarray[float, Any]: Taus
        """
        assert self._tau is not None
        return self._tau

    @property
    def q(self) -> np.ndarray[float, Any]:
        """q values (exponents)

        Returns:
            float: qs
        """
        assert self._q is not None
        return self._q

    # Statich methods

    @staticmethod
    def divisors(n: int) -> np.ndarray[int, Any]:
        """Compute divisors

        Args:
            n (int): number to compute divisors

        .. note::
            The choice for the scale factor is arbitrary and different scales will give different outcomes.
            Using the divisors is *time agnostic**.
            An alternative might be to choose a different range of scales according to the time horizon of the series.
            E.g., for daily data: `[1, 5, 10, 15, 21, 63, 126, 252]`,
            that is daily, weekly, bi-weekly, three-weekly, month, quarter, half-year, year.

        Returns:
            np.ndarray[int]: divisors
        """
        divs = [1]
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                divs.extend([i, int(n / i)])
        divs.extend([n])
        return np.array(sorted(set(divs)))[:-2]

    @staticmethod
    def adf_test(timeseries: pd.Series | np.ndarray, conf_level: float = 0.05) -> bool:
        """Augmented Dickey-Fuller test
        Wrapper aorund statsmodels.tsa.stattools.adfuller

        Args:
            timeseries (pd.Series | np.ndarray): the series to analyze
            conf_level (float, optional): confidence level for p-value. Defaults to 0.05.

        Returns:
            bool:  the series is strict stationary
        """
        print("Results of Dickey-Fuller Test:")
        test = adfuller(timeseries, autolag="AIC")
        output = pd.Series(
            test[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in test[4].items(): # type: ignore
            output["Critical Value (%s)" % key] = value
        print(output)
        return output["p-value"] < conf_level

    @staticmethod
    def kpss_test(timeseries: pd.Series | np.ndarray, conf_level: float = 0.05) -> bool:
        """Kwiatkowski-Phillips-Schmidt-Shin test for stationarity
        Wrapper aorund statsmodels.tsa.stattools.kpss

        Args:
            timeseries (pd.Series | np.ndarray): the series to analyze
            conf_level (float, optional): confidence level for p-value. Defaults to 0.05.

        Returns:
            bool: the series is NOT trend stationary
        """
        print("Results of KPSS Test:")
        test = kpss(timeseries, regression="c", nlags="auto")
        output = pd.Series(test[0:3], index=["Test Statistic", "p-value", "Lags Used"])
        for key, value in test[3].items():
            output["Critical Value (%s)" % key] = value
        print(output)
        return output["p-value"] < conf_level

    @staticmethod
    @njit(cache=True, parallel=True, fastmath=True)
    def compute_p(
        mul: np.ndarray[float, Any],
        S0: float,
        n: int = 30,
        num_sim: int = 10_000,
        seed: int = 1968,
    ) -> np.ndarray[float, Any]:
        """Compute geometric MMAR using Numba

        Args:
            mul (np.ndarray[float]): _description_
            S0 (float): initial price
            n (int, optional): number of steps. Defaults to 30.
            num_sim (int, optional): number of simulations. Defaults to 10_000.
            seed (int, optional): seed. Defaults to 1968.

        .. math::

            S(t) = S_{0} e^{B_{H}[\\theta(t)]}
        Returns:
            np.ndarray[float]: the simulated MMAR series
        """
        np.random.seed(seed)
        y = normal(loc=0, scale=1, size=(n, num_sim)).T
        x = np.empty(y.shape)
        for i in prange(x.shape[0]):
            x[i] = np.cumsum(mul * y[i])
        p = S0 * np.exp(x)
        return p

    # Utility methods

    def check_stationarity(self, conf_level: float = 0.05) -> None:
        """Check if the series is stationary

        Args:
            conf_level (float, optional): confidence level. Defaults to 0.05.

        See [statsmodels](https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html):
            * Case 1: Both tests conclude that the series is not stationary - The series is not stationary
            * Case 2: Both tests conclude that the series is stationary - The series is stationary
            * Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
            * Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.

        Returns:
            None
        """
        timeseries = np.diff(self._log_prices)
        adf = self.adf_test(timeseries, conf_level)
        print()
        kp = self.kpss_test(timeseries, conf_level)
        print()
        if (
            adf
        ):  # We can reject the null hypotesis of non stationarity, hence stationary
            if kp:  # For kpss is the opposite
                print("The time series is difference stationary")
            else:
                print("The time series is stationary")
        else:
            if kp:  # For kpss is the opposite
                print("The time series is not stationary")
            else:
                print("The time series is rend stationary")
        return None

    def check_normality(self, conf_level: float = 0.05) -> None:
        """Test distribution for Normality assumption

        Args:
            conf_level (float, optional): confidence level. Defaults to 0.05.
        """
        timeseries = np.diff(self._log_prices)
        test = jarque_bera(timeseries)
        if test.pvalue < conf_level: # type: ignore
            print("The time series is not Normal distributed")
        else:
            print("The time series is Normal distributed")
        return

    def check_autocorrelation(self, conf_level: float = 0.05, lags: int = 10) -> None:
        """Test series for autocorrelation

        Args:
            conf_level (float, optional): confidence level. Defaults to 0.05.
            lags (int, optional): umber of lags to consider. Defaults to 10.
        """
        timeseries = np.diff(self._log_prices)
        test = sm.stats.acorr_ljungbox(timeseries, lags=[lags])
        if cast(float, test.loc[lags, "lb_pvalue"]) < conf_level:
            print(
                "The time series is not independently distributed and has serial auto-correlation"
            )
        else:
            print("The time series is independently distributed")
        return

    def tauf(self, x: float) -> float:
        """Return tau(x) via interpolation

        Args:
            x (float): value for which compute tau

        .. math::

            \\tau(x)

        Returns:
            float: tau(x)
        """
        return cast(float,np.interp(x, xp=self.q, fp=self.tau))

    def legendre(self, alpha: float) -> float:
        """Compute Legendre transformation

        Args:
            alpha (float): estimation point

        Returns:
            float: transformed value
        """

        def F(x):
            return alpha * x - self.tauf(x)

        res = minimize_scalar(F, bounds=(self.q[0], self.q[-1]))
        xs = res.x
        return alpha * xs - self.tauf(xs)

    def check_hurst(self) -> None:
        """Check Hurst exponent

        Returns:
            None
        """
        if self.q is None:
            self.get_scaling()
        idx = np.argmax(self.q > 0)
        print(
            f"The hurst exponent must be {self.q[idx]:.2f} < Hurst < {self.q[idx-1]:.2f}"
        )
        return None

    # MMAR methods

    def get_alpha_min(self) -> float:
        """Compute alpha zero
        The smallest alpha for which the multifractal spectrum is defined.

        .. math::

            \\alpha_{0} = \\frac {\\tau(1.0)-\\tau(0.9999)}{q(1.0)-q(0.9999)}

        Returns:
            float: alpha zero
        """
        q = np.array([100, 99.99])
        tau = np.zeros_like(q)
        n = len(self.price)
        log_prices = self._log_prices
        t = self.divisors(n)
        delta = np.log(t)
        log_n = np.log(n)
        for r, qq in enumerate(q):
            y = np.zeros(len(t))
            # Partiton function
            for s, tt in enumerate(t):
                x = np.arange(0, n, tt)
                log_price_diff = np.diff(log_prices[x])
                y[s] = np.log(np.sum(np.abs(log_price_diff) ** qq)) - log_n
            # Estimate tau with linear regression
            lm_result = np.polyfit(delta, y, 1)
            tau[r] = lm_result[0]
        self._alpha_min = (tau[0] - tau[1]) / (q[0] - q[1])
        return self._alpha_min

    def get_scaling(
        self,
    ) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any], np.ndarray[float, Any]]:
        """Compute scaling function

        Returns:
            tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]: Taus, Cs, qs
        """
        q = np.linspace(0.01, 10.01, 1_000)
        tau = np.zeros_like(q)
        c = np.zeros_like(q)
        n = len(self.price)
        log_prices = self._log_prices
        t = self.divisors(n)
        delta = np.log(t)
        log_n = np.log(n)
        # Partiton function
        for r, qq in enumerate(q):
            y = np.zeros(len(t))
            for s, tt in enumerate(t):
                x = np.arange(0, n, tt)
                log_price_diff = np.diff(log_prices[x])
                y[s] = np.log(np.sum(np.abs(log_price_diff) ** qq)) - log_n
            # Estimate tau with linear regression
            lm_result = np.polyfit(delta, y, 1)
            tau[r] = lm_result[0]
            c[r] = lm_result[1]
        self._tau = np.concatenate(([-1], tau))
        self._q = np.concatenate(([0], q))
        self._c = np.concatenate(([1], c))
        return self._tau, self._c, self._q

    def get_hurst(self) -> float:
        """Compute Hurst exponent

        .. math::

            H = \\frac {1}{\\tau(q) =0}

        Returns:
            float: Hurst exponent
        """
        zero = root_scalar(self.tauf, bracket=[0.1, 4.9]).root
        self._H = 1 / zero
        return self._H

    def get_params(self) -> tuple[np.ndarray[float, Any], float]:
        """Compute main parameters for the MMAR model

        Returns:
            tuple[np.ndarray[float], float]: Theta values, returns volatility
        """
        alpha = np.arange(0.001, 1.101, 0.001)
        spectr = np.array([self.legendre(x) for x in alpha])
        self._m = cast(float,alpha[np.where(spectr < 0.99)[-1][-1]])
        b = 2
        K = int(np.log2(len(self.price)))
        self._mu = self.m / self.H
        self._sigma = np.sqrt((2 * (self.mu - 1)) / np.log(b))

        if self._volume is None:
            np.random.seed(self.seed)

            U = np.zeros((K, b))

            for kk in range(K):
                U[kk - 1, :] = b ** (-normal(size=b, loc=self._mu, scale=self._sigma))

            m1 = U[0, :].reshape(-1, 1)
            for i in range(1, K):
                m2 = np.concatenate([j * U[i, :].reshape(-1, 1) for j in m1], axis=1)
                m1 = m2

            self._theta = m2.flatten()
        else:
            volume = self._volume[-np.power(b, K) :]
            self._theta = volume / np.sum(volume)

        ret = np.diff(self._log_prices)
        self._sigma_ret = cast(float,np.std(ret))

        return self._theta, self._sigma_ret

    def config(self) -> None:
        """Run main fucntions for complete class configuration"""
        if self._alpha_min is None:
            self.get_alpha_min()
        if self._q is None:
            self.get_scaling()
        if self._H is None:
            self.get_hurst()
        if self._m is None:
            self.get_params()

    def get_MMAR_MC(
        self, S0: float, n: int = 30, num_sim: int = 10_000, seed: int = 1968
    ) -> np.ndarray[float, Any]:
        """Monte Carlo simulation accordig to MMAR model

        Args:
            S0 (float): initial price
            n (int, optional): number of steps. Defaults to 30.
            num_sim (int, optional): number of simulations. Defaults to 10_000.
            seed (int, optional): seed. Defaults to 1968.

        .. math::
            X(t,1) = \\underbrace {\\sqrt {b^k \\cdot \\theta_{k}(t)}}_{\\sigma(t)} \\cdot \\sigma \\cdot [B_{H}(t) -B_{H}(t-1)]

            \\text{mul } = \\sqrt {b^k \\cdot \\theta_{k}(t)} \\cdot \\sigma

        .. note::
            When the length of the simulation *n* is different than that of Theta, we should decide what values to use.
            In this case we decided to use the last values `theta[-n:]`, but maybe a random selection might be preferable.
            Something like `np.random.choice(theta, size=n, replace=False)`.

        Returns:
            np.ndarray[float, Any]: simulated data
        """
        b = 2
        theta = self.theta
        k = int(np.log2(len(theta)))
        sigma_ret = self.sigma_ret
        mul = np.sqrt(np.power(b, k) * theta[-n:]) * sigma_ret
        p = self.compute_p(mul, S0, n, num_sim, seed)
        return p

    # Plot methods

    def plot_scaling(self) -> None:
        """Plot the scaling function"""

        hypothetical_tau = 0.5 * self.q - 1
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].plot(
            self._q,
            hypothetical_tau,
            color="lightblue",
            linewidth=3,
            label="hypothetical monofractal",
        )
        ax[0].plot(
            self._q, self._tau, color="red", linewidth=3, label=r"observed $\tau $"
        )
        ax[0].set_xlabel("q")
        ax[0].set_ylabel(r"$\tau(q)$")
        ax[0].grid(True)
        ax[0].legend()
        ax[1].plot(
            self._q,
            hypothetical_tau,
            color="lightblue",
            linewidth=3,
            label="hypothetical monofractal",
        )
        ax[1].plot(self._q, self._c, color="red", linewidth=3, label="c")
        ax[1].set_xlabel("q")
        ax[1].set_ylabel(r"$\tau(q)$")
        ax[1].grid(True)
        ax[1].legend()
        plt.show()
        return

    def plot_qq(self) -> None:
        """QQ plot"""
        # Standardize returns
        ret_sd = zscore(self._log_prices)
        # QQ plot
        plt.figure(figsize=(8, 8))
        plt.title("QQ Plot of Standardized Returns")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Standardized Returns")
        plt.grid(True)
        plt.scatter(
            np.sort(np.random.normal(size=len(ret_sd))),
            np.sort(ret_sd),
            alpha=0.7,
            label="QQ Plot",
        )
        ypoints = plt.ylim()
        xpoints = plt.xlim()
        limits = (min(ypoints[0], xpoints[0]), max(ypoints[1], xpoints[1]))
        plt.ylim(limits)
        plt.xlim(limits)
        plt.plot(limits, limits, linestyle="-", color="r", lw=3)
        plt.legend()
        plt.show()
        return

    def plot_alpha(self) -> None:
        """Plot f(alpha)

        Returns:
            None
        """

        def f(x: float) -> float:
            """legendre transform

            Args:
                x (float): evaluation point

            Returns:
                float: transformed value
            """
            return np.min(x * self.q - self.tau)

        alpha = np.linspace(0.0, 1.0, num=len(self.q))
        ff = np.array([f(x) for x in alpha])
        plt.figure(figsize=(8, 8))
        plt.plot(alpha, ff, color="darkturquoise", linewidth=2, label=r"$f(\alpha)$")
        plt.grid(True)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$f(\alpha)$")
        plt.legend()
        plt.show()
        return

    def plot_alpha_theoretical(self) -> None:
        """Plot canonical f(alpha)

        .. math::
            f_X(\\alpha) = 1 - \\frac {(\\alpha - m_{\\alpha})^2} {4 \cdot H \cdot (m_{\\alpha}-H)}

        Returns:
            None
        """  # noqa: W605

        m = self.m
        H = self.H
        alpha = np.linspace(self.alpha_min, m, num=15)

        def f(x):
            return 1 - (x - m) ** 2 / (4 * H * (m - H))

        ff = np.array([f(x) for x in alpha])

        spectrum = np.array(
            [self.legendre(alpha_val) + self.tauf(alpha_val) for alpha_val in alpha]
        )

        plt.figure(figsize=(8, 8))
        plt.plot(alpha, ff, color="darkturquoise", linewidth=2, label=r"$f(\alpha)$")
        plt.plot(
            alpha,
            spectrum,
            marker="o",
            markerfacecolor="none",
            label="Theoretical path",
            linewidth=0,
        )
        plt.grid(True)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$f(\alpha)$")
        plt.legend()
        plt.show()
        return

    def plot_density(self) -> None:
        """Density plot"""
        ret = np.diff(self._log_prices)

        # Standardize returns
        ret_sd = (ret - np.mean(ret, axis=0, keepdims=True)) / np.std(
            ret, axis=0, ddof=1, keepdims=True
        )

        # Plot density and standard normal distribution
        x = np.arange(-5, 5, 0.1)

        plt.figure(figsize=(16, 10))
        plt.plot(x, norm.pdf(x), "k-", lw=2, label="Standard Normal Distribution")
        plt.hist(
            ret_sd,
            density=True,
            alpha=0.7,
            color="mediumspringgreen",
            label="Density of Standardized Returns",
            bins=30,
        )
        plt.grid(True)
        plt.xlabel("Standardized Returns")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        return
