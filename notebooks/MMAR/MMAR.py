from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import root_scalar
from numpy.random import normal
import pandas as pd



class MMAR:
    def __init__(self, price:pd.DataFrame):
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
        
        self._H = None

    @property
    def theta(self):
        return self._theta

    @property
    def H(self):
        return self._H

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    @property
    def sigma_ret(self):
        return self._sigma_ret

    @property
    def alpha_min(self):
        return self._alpha_min

    
    def tauf(self, x):
        return np.interp(x, xp=self._q, fp=self._tau)
    
    def legendre(self, alpha):
        def F(x):
            return alpha * x - self.tauf(x)
    
        res = minimize_scalar(F, bounds=(self._q[0], self._q[-1]))
        xs = res.x
        return alpha * xs - self.tauf(xs)

    def get_alpha_min(self):
        q = np.array([100,99.99])
        t = np.array([2,3,4,5,10,15,20])
        tau = np.zeros_like(q)
        N = len(self.price)
        log_prices = self._log_prices
        delta = np.log(t)
        log_n = np.log(N)
        r=0
        for qq in q:
            y = np.zeros(len(t))
            s = 0
            for tt in t:
                x = np.arange(0, N, tt)
                log_price_diff = log_prices[x[1:]] - log_prices[x[:-1]]
                y[s] = np.log(np.sum(np.abs(log_price_diff)**qq) ) - log_n
                s += 1
            lm_result = np.polyfit(delta, y, 1)
            tau[r]= lm_result[0]
            r+=1
        self._alpha_min = (tau[0] - tau[1]) / (q[0] - q[1])
        return self._alpha_min


    def get_scaling(self):
        q = np.linspace(0.01,10,1_000)
        t = np.array([2,3,4,5,10, 15,20])
        tau = np.zeros_like(q)
        c = np.zeros_like(q)
        N = len(self.price)
        log_prices = self._log_prices
        delta = np.log(t)
        log_n = np.log(N)
        r=0
        for qq in q:
            y = np.zeros(len(t))
            s = 0
            for tt in t:
                x = np.arange(0, N, tt)
                log_price_diff = log_prices[x[1:]] - log_prices[x[:-1]]
                y[s] = np.log(np.sum(np.abs(log_price_diff)**qq) ) - log_n
                s += 1
            lm_result = np.polyfit(delta, y, 1)
            tau[r]= lm_result[0]
            c[r]= lm_result[1]
            r+=1
        self._tau = np.concatenate(([-1], tau))
        self._q = np.concatenate(([0], q))
        self._c = np.concatenate(([1], c))
        return self._tau, self._c, self._q


    def plot_scaling(self):
        self.config()
        bm = 0.5 * self._q - 1
        fig, ax = plt.subplots(1,2, figsize=(18,6))
        ax[0].plot(self._q, bm, color='lightblue', linewidth=3, label='bm')
        ax[0].plot(self._q, self._tau, color='purple', linewidth=3, label='tau')
        ax[0].set_xlabel("q")
        ax[0].set_ylabel(r"$\tau(q)$")
        ax[0].legend()
        ax[1].plot(self._q, bm, color='lightblue', linewidth=3, label='bm')
        ax[1].plot(self._q, self._c, color='red', linewidth=3, label='c')
        ax[1].set_xlabel("q")
        ax[1].set_ylabel(r"$\tau(q)$")
        ax[1].legend()
    
        plt.show()
        return 

    def get_hurst(self):
        zero = root_scalar(self.tauf, bracket=[1.7, 2.91]).root
        self._H = 1/zero
        return self._H


    def plot_alpha(self):
        self.config()
            
        m = self._m
        H = self._H
        alpha = np.linspace(self._alpha_min, m, num=15)
        
        f = lambda x: 1 - (x - m)**2 / (4 * H * (m - H))
        ff = np.array([f(x) for x in alpha])
        
        plt.plot(alpha, ff, color='darkturquoise', linewidth=2, label='f')
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$f(\alpha)$")
        plt.legend()
        plt.show()
        return 

    def get_params(self):            
        alpha = np.arange(self._alpha_min, 1.101, 0.001)
        spectr = np.array([self.legendre(x)  for x in alpha])
        self._m = alpha[np.where(spectr < 0.99)[-1][-1]]
        b = 2
        k = np.arange(1, 12)
        
        self._mu = self._m / self._H
        self._sigma = np.sqrt((2 * (self._mu - 1)) / np.log(b))
        U = np.zeros((len(k), b))

        for kk in k:
            U[kk - 1, :] = b**(-normal(size=b, loc=self._mu, scale=self._sigma))
        
        m1 = U[0, :].reshape(-1, 1)
        for i in range(1, len(k)):
            m2 = np.concatenate([j * U[i, :].reshape(-1, 1) for j in m1], axis=1)
            m1 = m2
        
        self._theta = m2.flatten()
        
        ret = np.diff(np.log(self.price.values))
        self._sigma_ret = np.sqrt(np.var(ret))

        return self._theta, self._sigma_ret

    def config(self):
        if self._alpha_min is None:
            self.get_alpha_min()
        if self._q is None:
            self.get_scaling()
        if self._H is None:
            self.get_hurst()
        if self._m is None:
            self.get_params()
        

    def get_MMAR_MC(self, S0:float, n:int=30, num_sim:int=10_000)->np.ndarray:
        self.config()
        b = 2
        k = np.arange(1, 12)
        theta = self._theta
        sigma_ret = self._sigma_ret
        x = np.cumsum(np.sqrt(b**k[-1] * theta[:n]) * sigma_ret * norm.rvs(size=(n, num_sim)).T, axis =1)
        p = S0 * np.exp(x)
        return p