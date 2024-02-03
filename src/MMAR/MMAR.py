from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, zscore, jarque_bera
from scipy.optimize import root_scalar
from numpy.random import normal
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from numba import njit, prange



class MMAR:
    def __init__(
        self, price: pd.Series, seed: int = 42, volume: pd.Series | None = None
    )->None:
        """_summary_

        Args:
            price (pd.Series): _description_
            seed (int, optional): _description_. Defaults to 42.
            volume (pd.Series | None, optional): _description_. Defaults to None.

        Returns:
            None: _description_
        """
        self.price = price
        self._log_prices = np.log(price.values)
        self._theta = None
        self._m = None
        self._sigma = None
        self._sigma_ret = None
        self._mu = None
        self._q = None
        self._c = None
        self._tau = None
        self._alpha_min = None
        self.seed = seed
        if volume is not None:
            self._volume = volume.values
        else:
            self._volume = volume
        self._H = None
        self.post_init()

    def post_init(self)->None:
        """_summary_
        """
        n = len(self.price)
        m = n
        while len(self.divisors(m)) < 5:
            m -= 1
        if m != n:
            print(
                f"The series has been adjusted: the orginal size was {n}, the new size is {m} with {len(self.divisors(m))} dividers."
            )
            self.price = self.price[-m:]
            self._log_prices = np.log(self.price.values)

    # Properties
    @property
    def theta(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._theta

    @property
    def H(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._H

    @property
    def mu(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._mu

    @property
    def sigma(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._sigma

    @property
    def sigma_ret(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._sigma_ret

    @property
    def alpha_min(self)->float:
        """_summary_

        Returns:
            float: _description_
        """
        return self._alpha_min

    @property
    def tau(self)->float:
        """_summary_

        Returns:
            float: _description_
        """
        return self._tau

    @property
    def q(self)->float:
        """_summary_

        Returns:
            float: _description_
        """
        return self._q

    # Statich methods

    @staticmethod
    def divisors(n:int)->np.ndarray[int]:
        """_summary_

        Args:
            n (int): _description_

        Returns:
            np.ndarray[int]: _description_
        """
        divs = [1]
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                divs.extend([i, int(n / i)])
        divs.extend([n])
        return np.array(sorted(list(set(divs))))[:-2]

    @staticmethod
    def adf_test(timeseries:pd.Series|np.ndarray, conf_level: float = 0.05)->bool:
        """_summary_

        Args:
            timeseries (pd.Series | np.ndarray): _description_
            conf_level (float, optional): _description_. Defaults to 0.05.

        Returns:
            bool: _description_
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
        for key, value in test[4].items():
            output["Critical Value (%s)" % key] = value
        print(output)
        return output["p-value"] < conf_level

    @staticmethod
    def kpss_test(timeseries:pd.Series|np.ndarray, conf_level: float = 0.05)->bool:
        """_summary_

        Args:
            timeseries (pd.Series | np.ndarray): _description_
            conf_level (float, optional): _description_. Defaults to 0.05.

        Returns:
            bool: _description_
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
    def compute_p(mul:np.ndarray[float], S0: float, n: int = 30, num_sim: int = 10_000,  seed: int = 1968)->np.ndarray[float]:
        """_summary_

        Args:
            mul (np.ndarray[float]): _description_
            S0 (float): _description_
            n (int, optional): _description_. Defaults to 30.
            num_sim (int, optional): _description_. Defaults to 10_000.
            seed (int, optional): _description_. Defaults to 1968.

        Returns:
            np.ndarray[float]: _description_
        """
        np.random.seed(seed)
        y = normal(loc=0, scale=1, size=(n, num_sim)).T
        x = np.empty(y.shape)
        for i in prange(x.shape[0]):
            x[i] = np.cumsum(
                    mul
                    * y[i]
                )
        p = S0 * np.exp(x)
        return p

    # Utility methods

    def check_stationarity(self, conf_level: float = 0.05)->None:
        """_summary_

        Args:
            conf_level (float, optional): _description_. Defaults to 0.05.

        Returns:
            _type_: _description_
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

    def check_normality(self, conf_level: float = 0.05)->None:
        """_summary_

        Args:
            conf_level (float, optional): _description_. Defaults to 0.05.
        """
        timeseries = np.diff(self._log_prices)
        test = jarque_bera(timeseries)
        if test.pvalue < conf_level:
            print("The time series is not Normal distributed")
        else:
            print("The time series is Normal distributed")
        return

    def check_autocorrelation(self, conf_level: float = 0.05, lags: int = 10)->None:
        """_summary_

        Args:
            conf_level (float, optional): _description_. Defaults to 0.05.
            lags (int, optional): _description_. Defaults to 10.
        """
        timeseries = np.diff(self._log_prices)
        test = sm.stats.acorr_ljungbox(timeseries, lags=[lags])
        if test.loc[lags, "lb_pvalue"] < conf_level:
            print(
                "The time series is not independently distributed and has serial auto-correlation"
            )
        else:
            print("The time series is independently distributed")
        return

    def tauf(self, x, q, tau)->float:
        """_summary_

        Args:
            x (_type_): _description_
            q (_type_): _description_
            tau (_type_): _description_

        Returns:
            float: _description_
        """
        return np.interp(x, xp=q, fp=tau)

    def legendre(self, alpha, q, tau):
        def F(x):
            return alpha * x - self.tauf(x, q, tau)

        res = minimize_scalar(F, bounds=(q[0], q[-1]))
        xs = res.x
        return alpha * xs - self.tauf(xs, q, tau)

    def check_hurst(self)->None:
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.q is None:
            self.get_scaling()
        idx = np.argmax(self.q > 0)
        print(
            f"The hurst exponent must be {self.q[idx]:.2f} < Hurst < {self.q[idx-1]:.2f}"
        )
        return None

    # MMAR methods

    def get_alpha_min(self)->float:
        """_summary_

        Returns:
            _type_: _description_
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

    def get_scaling(self)->tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """_summary_

        Returns:
            tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]: _description_
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

    def get_hurst(self)->float:
        """_summary_

        Returns:
            float: _description_
        """
        zero = root_scalar(self.tauf, args=(self.q, self.tau), bracket=[0.1, 4.9]).root
        self._H = 1 / zero
        return self._H

    def get_params(self, K: int = 12)-> tuple[np.ndarray[float], float]:
        """_summary_

        Args:
            K (int, optional): _description_. Defaults to 12.

        Returns:
            tuple[np.ndarray[float], float]: _description_
        """
        alpha = np.arange(0.001, 1.101, 0.001)
        spectr = np.array([self.legendre(x, self.q, self.tau) for x in alpha])
        self._m = alpha[np.where(spectr < 0.99)[-1][-1]]
        b = 2

        self._mu = self._m / self.H
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
            volume = self._volume[-np.power(b, K - 1) :]
            self._theta = volume / np.sum(volume)

        ret = np.diff(self._log_prices)
        self._sigma_ret = np.std(ret)

        return self._theta, self._sigma_ret

    def config(self)-> None:
        """_summary_
        """
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
    ) -> np.ndarray:
        self.config()
        b = 2
        theta = self.theta
        k = int(np.log2(len(theta)))
        sigma_ret = self.sigma_ret
        mul = np.sqrt(np.power(2,k) * theta[-n:])* sigma_ret
        p = self.compute_p(mul, S0, n, num_sim, seed)
        return p

    # Plot methods

    def plot_scaling(self)->None:
        """_summary_
        """
        self.config()
        hypothetical_tau = 0.5 * self._q - 1
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

    def plot_qq(self)->None:
        """_summary_
        """
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

    def plot_alpha(self)->None:
        """_summary_

        Returns:
            _type_: _description_
        """
        self.config()

        m = self._m
        H = self.H
        alpha = np.linspace(self.alpha_min, m, num=15)

        def f(x):
            return 1 - (x - m) ** 2 / (4 * H * (m - H))

        ff = np.array([f(x) for x in alpha])

        plt.plot(alpha, ff, color="darkturquoise", linewidth=2, label="f")
        plt.grid(True)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$f(\alpha)$")
        plt.legend()
        plt.show()
        return

    def plot_density(self) -> None:
        """_summary_
        """
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
