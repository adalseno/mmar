#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:16:19 2023

version 1.0.
version 1.1, changes name of these function, 20230124.
version 1.2, changes AWC and VVL method, 20230127.
version 1.3, add RSAnalysis method function, 20230510.
version 1.4, add LSSD, LSV method and Local Whittle function, 20230621.

@author: zqfeng
"""

import pywt
import numpy as np
from .fgn import fgn
from math import pi, gamma
from scipy import fft, stats
from .AddMethods import AddMethods


class HurstIndexSolver(AddMethods):

    def __init__(self):
        None

    def __FitCurve(self, Scale: list, StatisticModel: list,
                   method='L2') -> float:
        """
        Fitting scale ~ statisticModel in a log-log plot.
        """
        Scale = np.log10(np.array([Scale]))
        Scale = np.vstack([Scale, np.ones(len(StatisticModel))]).T
        if method == 'L2':
            # slope = np.polyfit(np.log10(Cm), np.log10(AM), 1)[0]
            slope, c = np.linalg.lstsq(
                Scale,
                np.log10(StatisticModel),
                rcond=-1
            )[0]
        elif method == 'L1':
            slope, c = super().OLE_linprog(
                Scale,
                np.array([np.log10(StatisticModel)]).T
            )
            slope = slope[0]
        return slope

    def EstHurstClustering(self, ts, order: float, minimal=10,
                           method='L2') -> float:
        """
        Calculate the Hurst exponent using Clustering Method.

        Parameters
        ----------
        ts     : Time series.
        minimal: The box sizes that the sample is divided into, default as 10.
        method : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        The Hurst exponent of time series X using
        Absolute Moments Method (AM).

        Reference
        ---------
        Hamza A H, Hmood M Y. Comparison of Hurst exponent estimation methods
        [J]. 2021.

        written by z.q.feng at 2022.09.05
        """
        N = len(ts)
        # make sure m is large and (N / m) is large
        OptN = super().findOptN(N, minimal=minimal)
        M = super().Divisors(OptN, minimal=minimal)

        ts = ts[N - OptN:]
        # The mean for series
        Avg = np.mean(ts)

        CM = []
        for m in M:
            k = OptN // m
            # remove the redundant data at the begin
            # each row is a subseries with N m
            Xsub = np.reshape(ts, [k, m])
            # mean of each suseries
            Xm = np.mean(Xsub, axis=1)
            # order == 1 : Absolute Moments Method
            # order == 2 : Aggregated Variance Method
            CM.append(np.mean(abs(Xm - Avg)**order))

        slope = self.__FitCurve(M, CM, method=method)
        return slope / order + 1

    def EstHurstAbsoluteMoments(self, ts, minimal=20, method='L2') -> float:
        """
        Calculate the Hurst exponent using Cluster Method.

        Parameters
        ----------
        ts     : Time series.
        minimal: The box sizes that the sample is divided into, default as 10.
        method : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        The Hurst exponent of time series X using
        Absolute Moments Method (AM).

        Reference
        ---------
        Hamza A H, Hmood M Y. Comparison of Hurst exponent estimation methods
        [J]. 2021.

        written by z.q.feng at 2022.09.05
        """
        N = len(ts)
        # make sure m is large and (N / m) is large
        OptN = super().findOptN(N, minimal=minimal)
        M = super().Divisors(OptN, minimal=minimal)

        ts = ts[N - OptN:]
        # The mean for series
        Avg = np.mean(ts)

        AM = []
        for m in M:
            k = OptN // m
            # remove the redundant data at the begin
            # each row is a subseries with N m
            Xsub = np.reshape(ts, [k, m])
            # mean of each suseries
            Xm = np.mean(Xsub, axis=1)
            AM.append(np.linalg.norm(Xm - Avg, 1) / len(Xm))

        slope = self.__FitCurve(M, AM, method=method)
        return slope + 1

    def EstHurstAggregateVariance(self, ts, minimal=20, method='L2') -> float:
        """
        Calculate the Hurst exponent using Cluster Method.

        Parameters
        ----------
        ts     : Time series.
        minimal: The box sizes that the sample is divided into, default as 10.
        method : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        The Hurst exponent of time series X using
        Aggregate Variance Method (AV).

        Reference
        ---------
        Hamza A H, Hmood M Y. Comparison of Hurst exponent estimation methods
        [J]. 2021.

        written by z.q.feng at 2022.09.05
        """
        N = len(ts)
        # The mean for series
        # Avg = np.mean(ts)

        # make sure m is large and (N / m) is large
        OptN = super().findOptN(N, minimal=minimal)
        M = super().Divisors(OptN, minimal=minimal)

        AV = []
        for m in M:
            k = OptN // m
            # remove the redundant data at the begin
            # each row is a subseries with N m
            Xsub = np.reshape(ts[N - OptN:], [k, m])
            # mean of each suseries
            Xm = np.mean(Xsub, axis=1)
            AV.append(np.var(Xm, ddof=0))
            # AV.append(np.var(Xm - Avg, ddof=1))

        slope = self.__FitCurve(M, AV, method=method)
        return slope / 2 + 1

    def EstHurstDFAnalysis(self, ts, minimal=20, method='L2') -> float:
        """
        DFA Calculate the Hurst exponent using DFA analysis.

        Parameters
        ----------
        ts     : Time series.
        minimal: The box sizes that the sample is divided into, default as 10.
        method : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        The Hurst exponent of time series X using
        Detrended Fluctuation Analysis (DFA).

        References
        ----------
        [1] C.-K.Peng et al. (1994) Mosaic organization of DNA nucleotides,
        Physical Review E 49(2), 1685-1689.
        [2] R.Weron (2002) Estimating long range dependence: finite sample
        properties and confidence intervals, Physica A 312, 285-299.

        Written by z.q.feng (2022.09.23).
        Based on dfa.m orginally written by afal Weron (2011.09.30).
        """
        DF = []
        N = len(ts)
        y = np.cumsum(ts - np.mean(ts))

        OptN = super().findOptN(len(ts), minimal=minimal)
        M = super().Divisors(OptN, minimal=minimal)

        for m in M:
            k = OptN // m
            Y = np.reshape(y[N - OptN:], [m, k], order='F')
            F = np.copy(Y)
            # t = 1, 2, ..., m
            t = np.linspace(1, m, m)
            for i in range(k):
                p = np.polyfit(t, Y[:, i], 1)
                F[:, i] = Y[:, i] - t * p[0] - p[1]
            DF.append(np.mean(np.std(F)))

        slope = self.__FitCurve(M, DF, method=method)
        return slope

    def __getBox(self, j: int) -> int:
        """
        [2^{(j-1)/4}] for j in (11, 12, 13, ...) if k > 4
        """
        if j < 5:
            return j
        else:
            return int(2 ** ((j + 5) / 4))

    def EstHurstHiguchi(self, ts, minimal=11, method='L2') -> float:
        """
        Calculate the Hurst exponent using Higuchi Method.

        Parameters
        ----------
        ts     : Time series.
        method : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        The Hurst exponent of time series X using Higuchi Method (HM).

        References
        ----------
        [1] Higuchi T. Approach to an irregular time series on the basis of
            the fractal theory[J]. Physica D: Nonlinear Phenomena, 1988, 31(2):
            277-283.
        """
        N = len(ts)
        Lm, Cm = [], []
        # FGN --diff--> Gaussian
        Y = np.cumsum(ts - np.mean(ts))

        for j in range(1, minimal):
            Lk = []
            m = self.__getBox(j)
            Cm.append(m)
            k = N // m
            Xsub = np.reshape(Y[N % m:], [k, m])
            for i in range(1, k):
                Lk.append(abs(Xsub[i] - Xsub[i - 1]))
            # Lm = np.mean(np.array(Lk), axis=0) * (N - 1) / k / k
            Lm.append(np.mean(Lk) * (N - 1) / m / m)

        slope = self.__FitCurve(Cm, Lm, method=method)
        return slope + 2

    def EstHurstRegrResid(self, ts, minimal=20, method='L2') -> float:
        '''
        Variance of the regression residuals (VRR) for Hurst Index.
        '''
        Sigma = []
        N = len(ts)

        OptN = super().findOptN(N, minimal=minimal)
        M = super().Divisors(OptN, minimal=minimal)[:-3]

        for m in M:
            res = []
            k = OptN // m
            t = np.linspace(1, m, m)
            X = np.reshape(ts[N - OptN:], [k, m])
            for i in range(k):
                Y = np.cumsum(X[i] - np.mean(X[i]))
                a, b = np.polyfit(t, Y, 1)
                res.append(np.std(Y - a * t - b, ddof=1))
            Sigma.append(np.mean(res))

        slope = self.__FitCurve(M, Sigma, method=method)
        return slope

    def __HalfSeries(self, s: list, n: int) -> list:
        X = []
        for i in range(0, len(s) - 1, 2):
            X.append((s[i] + s[i + 1]) / 2)
        # if length(s) is odd
        if len(s) % 2 != 0:
            X.append(s[-1])
            n = (n - 1) // 2
        else:
            n = n // 2
        return [np.array(X), n]

    def RS4Hurst(self, ts: np.ndarray, minimal=4, method='L2') -> float:
        """
        RS Analysis for solve the Hurst exponent.
        """
        ts = np.array(ts)
        # N is use for storge the length sequence
        N, RS, n = [], [], len(ts)
        while (True):
            N.append(n)
            # Calculate the average value of the series
            m = np.mean(ts)
            # Construct mean adjustment sequence
            mean_adj = ts - m
            # Construct cumulative deviation sequence
            cumulative_dvi = np.cumsum(mean_adj)
            # Calculate sequence range
            srange = max(cumulative_dvi) - min(cumulative_dvi)
            # Calculate the unbiased standard deviation of this sequence
            unbiased_std_dvi = np.std(ts, ddof=1)
            # Calculate the rescaled range of this sequence
            RS.append(srange / unbiased_std_dvi)
            # While n < 2 then break
            if n < minimal:
                break
            # Rebuild this sequence by half length
            ts, n = self.__HalfSeries(ts, n)
        # Get Hurst-index by fit log(RS)~log(n)
        slope = self.__FitCurve(N, RS, method=method)
        return slope

    def EstHurstRSAnalysis(self, ts, minimal=20, IsRandom=False,
                           method='L2') -> float:
        '''
        RS Calculate the Hurst exponent using DFA analysis.

        Parameters
        ----------
        ts     : Time series.
        minimal: The box sizes that the sample is divided into, default as 50.
        method : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        The Hurst exponent of time series X using
        Rescaled Range Analysis (DFA).

        References
        ----------
        [1] H.-H.Amjad et al. (2021) Comparison of Hurst exponent estimation
        methods, Physical Review E 49(2), 1685-1689.
        [2] R.Weron (2002) Estimating long range dependence: finite sample
        properties and confidence intervals, Physica A 312, 285-299.
        [3] E.E.Peters (1994) Fractal Market Analysis, Wiley.
        [4] A.A.Anis, E.H.Lloyd (1976) The expected value of the adjusted
        rescaled Hurst range of independent normal summands, Biometrica 63,
        283-298.

        Written by z.q.feng (2022.09.23).
        '''
        N = len(ts)

        OptN = super().findOptN(N, minimal=minimal)
        M = super().Divisors(OptN, minimal=minimal)

        # M is use for storge the length sequence
        RSe, ERS_AL = [], []
        for m in M:
            RSm = []
            k = OptN // m
            X = np.reshape(ts[N - OptN:], [k, m])
            for Xm in X:
                # Calculate the average value of the sub-series
                Em = np.mean(Xm)
                # Construct mean adjustment sequence
                mean_adj = Xm - Em
                # Construct cumulative deviation sequence
                cumulative_dvi = np.cumsum(mean_adj)
                # Calculate sequence range
                srange = max(cumulative_dvi) - min(cumulative_dvi)
                # Calculate the unbiased standard deviation of this sequence
                unbiased_std_dvi = np.std(mean_adj, ddof=1)
                # Calculate the rescaled range of this sequence under n length
                RSm.append(srange / unbiased_std_dvi)
            RSe.append(np.mean(RSm))

        # Compute Anis-Lloyd[4] and Peters[3] corrected theoretical E(R/S)
        for m in M:
            # (see [2] for details)
            K = np.linspace(1, m - 1, m - 1)
            ratio = (m - 0.5) / m * np.sum(((-K + m) / K)**0.5)
            if m > 340:
                ERS_AL.append(ratio / (0.5 * pi * m)**0.5)
            else:
                ERS_AL.append((gamma(0.5*(m-1))*ratio)/(gamma(0.5*m)*pi**0.5))
        # see Peters[3] page 66 eq 5.1
        ERS = (0.5 * pi * np.array(M))**0.5

        RSe, ERS_AL = np.array(RSe), np.array(ERS_AL)
        RS = RSe - ERS_AL + ERS if IsRandom else RSe

        slope = self.__FitCurve(M, RS, method=method)
        return slope

    def EstHurstRSAnalysis2(self, ts, minimal=20, method='L2') -> float:
        '''
        RS Calculate the Hurst exponent using DFA analysis.

        Parameters
        ----------
        ts     : Time series.
        minimal: The box sizes that the sample is divided into, default as 50.
        method : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        The Hurst exponent of time series X using
        Rescaled Range Analysis (DFA).

        References
        ----------
        [1] H.-H.Amjad et al. (2021) Comparison of Hurst exponent estimation
        methods, Physical Review E 49(2), 1685-1689.
        [2] R.Weron (2002) Estimating long range dependence: finite sample
        properties and confidence intervals, Physica A 312, 285-299.
        [3] E.E.Peters (1994) Fractal Market Analysis, Wiley.
        [4] A.A.Anis, E.H.Lloyd (1976) The expected value of the adjusted
        rescaled Hurst range of independent normal summands, Biometrica 63,
        283-298.

        Written by z.q.feng (2022.09.23).
        '''
        N = len(ts)

        OptN = super().findOptN(N, minimal=minimal)
        M = super().Divisors(OptN, minimal=minimal)
        y = np.cumsum(ts - np.mean(ts))

        # M is use for storge the length sequence
        RSe = []
        for m in M:
            RSm = []
            k = OptN // m
            X = np.reshape(y[N - OptN:], [k, m])
            for i in range(k):
                # Calculate sequence range
                srange = max(X[i]) - min(X[i])
                # Calculate the unbiased standard deviation of this sequence
                unbiased_std_dvi = np.std(ts[i*m:(i+1)*m], ddof=0)
                # Calculate the rescaled range of this sequence under n length
                RSm.append(srange / unbiased_std_dvi)
            RSe.append(np.mean(RSm))

        slope = self.__FitCurve(M, RSe, method=method)
        return slope

    def EstHurstPeriodogram(self, ts, cutoff=0.5, method='L2') -> float:
        """
        Parameters
        ----------
        ts     : Time series.
        cutoff : Level of low Fourier frequencies, default as 0.5.
        method : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        Hurst exponent H of a time series ts estimated using the
        Geweke-Porter-Hudak (GPH, 1983) spectral estimator for periods
        lower than max(period)^CUTOFF, where CUTOFF=0.5.

        References
        ----------
        [1] J.Geweke, S.Porter-Hudak (1983) The estimation and application of
        long memory time series models, Journal of Time Series Analysis 4,
        221-238.
        [2] R.Weron (2002) Estimating long range dependence: finite sample
        properties and confidence intervals, Physica A 312, 285-299.

        Written by z.q.feng (2022.09.21).
        """
        N = len(ts)
        # Compute the Fourier transform of the data
        # Remove the first component of Y, which is simply the sum of the data
        Y = fft.fft(ts)[1:N // 2 + 1]
        # Define the frequencies
        freq = np.linspace(1 / N, 0.5, N // 2)
        # Find the low Fourier frequencies
        index = freq < 1 / N ** cutoff
        # The periodogram is deﬁned as
        IL = 4 * np.sin(freq[index] / 2) ** 2
        # Compute the power as the squared absolute value of Y
        # A plot of power versus frequency is the 'periodogram'
        power = abs(Y[index]) ** 2 / N
        slope = self.__FitCurve(IL, power, method=method)
        return 0.5 - slope

    def EstHurstAWC(self, ts, wavelet="db24", wavemode="periodization",
                    method='L2') -> float:
        """
        Parameters
        ----------
        ts      : Time series.
        wavelet : Wavelet type, default as "db24".
        wavemode: The discrete wavelet transform extension mode,
                  default as "periodization".
        method  : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        Hurst exponent H of a time series ts estimated using the
        Average Wavelet Coefficient (AWC) method.

        References
        ----------
        [1] I.Simonsen, A.Hansen, O.Nes (1998) Determination of the Hurst
        exponent by use of wavelet transforms, Physical Review E 58, 2779-2787.
        [2] I.Simonsen (2003) Measuring anti-correlations in the Nordic
        electricity spot market by wavelets, Physica A 322, 597-606.
        [3] R.Weron, I.Simonsen, P.Wilman (2004) Modeling highly volatile and
        seasonal markets: evidence from the Nord Pool electricity market, in
        "The Application of Econophysics", ed. H. Takayasu, Springer, 182-191.
        [4] R.Weron (2006) Modeling and Forecasting Electricity Loads and
        Prices: A Statistical Approach, Wiley, Chichester.

        Written by z.q.feng (2022.09.24).
        Based on function awc_hurst.m originally written by Rafal Weron
        (2014.06.21).
        """
        # Do not allow for too large values of N
        N = int(np.floor(np.log2(len(ts))))
        # Daubechies wavelet of order 24
        # Set the DWT mode to periodization, see pywt.Modes for details
        # coeffs contains one Approximation and N Detail-Coefficients
        coeffs = pywt.wavedec(ts, wavelet=wavelet, mode=wavemode, level=N)
        sc, awc = [], []
        for i in range(1, N - 1):
            # Get the Detail-Coefficients
            cD = coeffs[-i]
            sc.append(2 ** i)
            # Compute the AWC statistics
            awc.append(np.mean(abs(cD)))
        # Level value of N is too high
        # sc, awc = sc[:-1], awc[:-1]
        slope = self.__FitCurve(sc, awc, method=method)
        return slope + 0.5

    def EstHurstVVL(self, ts, wavelet="haar", wavemode="periodization",
                    method='L2') -> float:
        """
        Parameters
        ----------
        ts      : Time series. Be careful of N >= 2^15.
        wavelet : Wavelet type, default as "db24".
        wavemode: The discrete wavelet transform extension mode,
                  default as "periodization".
        method  : The method to fit curve, default as minimal l2-norm.

        Returns
        -------
        Hurst exponent H of a time series ts estimated using the
        Variance Versus Level Method (VVL) using wavelets.

        References
        ----------
        [1] Hamza A H, Hmood M Y. Comparison of Hurst exponent estimation
        methods[J]. 2021.

        Written by z.q.feng (2022.09.24).
        """
        # Do not allow for too large values of N
        N = int(np.floor(np.log2(len(ts))))
        # Haar wavelet
        # Set the DWT mode to periodization, see pywt.Modes for details
        # coeffs contains one Approximation and N Detail-Coefficients
        coeffs = pywt.wavedec(ts, wavelet, level=N, mode=wavemode)
        sc, vvl = [], []
        for i in range(1, N - 1):
            # Get the Detail-Coefficients
            cD = coeffs[-i]
            sc.append(2 ** i)
            # Compute the VVL statistics
            vvl.append(np.var(abs(cD), ddof=1))
        slope = self.__FitCurve(sc, vvl, method=method)
        return (slope + 1) / 2

    def EstHurstLocalWhittle(self, ts: np.ndarray) -> float:
        """
        Semiparametric Gaussian Estimation via Fast Fourier transform.

        Parameters
        ----------
        ts        : Time series.
        iter_count: Accuracy for estimation degree.

        Returns
        -------
        The Hurst exponent of time series ts.

        References
        ----------
        [1] Robinson, P. M.(1995). "Gaussian semiparametric estimation
            of long-range dependence". The Annals of Statistics. 23(5):
            1630–1661. doi:10.1214/aos/1176324317.

        Written by z.q.feng (2023.06.07).
        """
        n = len(ts)
        m = n // 2  # Less than n / 2
        w = fft.fft(ts)[1:m + 1]  # Fast Fourier transform
        Periodogram = abs(w)**2
        Freqs = np.linspace(1, m + 1, m) / n  # Frequences

        def Whittle_target(H, **kwargs):
            """
            Target function to minimize in Local Whittle method.
            """
            Freqs = kwargs["Freqs"]
            Periodogram = kwargs["Periodogram"]
            gH = np.mean(Freqs**(2 * H - 1) * Periodogram)
            rH = np.log(gH) - (2 * H - 1) * np.mean(np.log(Freqs))
            return rH

        H = super().LocalMin(
                Whittle_target,
                [0.001, 0.999],
                Periodogram=Periodogram,
                Freqs=Freqs
            )

        return H

    def EstHurstLSSD(self, ts: np.ndarray, max_scale: int, p=6, q=0,
                     eps=1e-6) -> float:
        """
        Least Squares based on Standard Deviation. The fitting error is
        constructed by sample standard deviation.

        Parameters
        ----------
        ts        : Time series.
        max_scale : Maximum aggregation scale(>= length / 10).
        p         : Parameter used to determine the weights.
        q         : Parameter used to determine the penalty factor.
        eps       : Accuracy for estimation.

        Returns
        -------
        The Hurst exponent of time series ts.

        References
        ----------
        [1] Koutsoyiannis D (2003) Climate change, the Hurst phenomenon, and
            hydrological statistics. Hydrological Sciences Journal 48(1):3–24.
            doi:10.1623/hysj.48.1.3.43481.
        [2] Tyralis H, Koutsoyiannis D (2011) Simultaneous estimation of the
            parameters of the Hurst-Kolmogorov stochastic process. Stochastic
            Environmental Research & Risk Assessment 25(1):21–33.
            doi:10.1007/s004770100408x.

        Written by z.q.feng (2023.06.07).
        """
        n = len(ts)
        # This maximum value was chosen so that VarSeq can be estimated
        # from at least 10 data values.
        max_scale = min(max_scale, n // 10 + 1)
        stdSeq = np.linspace(1, max_scale, max_scale)
        # {1, 2, ..., max_scale}
        kscale = np.linspace(1, max_scale, max_scale).astype(int)
        # Unbiased Stadnard Deviation of each sample
        for scale in kscale:
            sample = np.sum(
                        np.reshape(ts[n % scale:], [n // scale, scale]),
                        axis=1
                        )
            stdSeq[scale - 1] = np.std(sample, ddof=1)

        def LSSDIterFun(H, **kwargs):
            """
            A constructive mapping in LSSD method. Each improved estimate can
            reduce the fitting error and continues this way until convergence.
            """
            n = kwargs["n"]
            p = kwargs["p"]
            q = kwargs["q"]
            kscale = kwargs["kscale"]
            stdSeq = kwargs["stdSeq"]
            f = n / kscale
            kp = kscale**p
            logk = np.log(kscale)
            # eq.A.3 in Ref[1]
            a1 = np.sum(1 / kp)
            a2 = np.sum(logk / kp)
            # eq.12 in Ref[2]
            ckH = ((f - f**(2 * H - 1)) / (f - 0.5))**0.5
            # eq.A.5 in Ref[1]
            dkH = logk + np.log(f) / (1 - f**(2 - 2 * H))
            # eq.A.4 in Ref[1]
            aH_1 = np.sum(dkH / kp)
            aH_2 = np.sum(dkH * logk / kp)
            bH_1 = np.sum((np.log(stdSeq) - np.log(ckH)) / kp)
            bH_2 = np.sum(dkH * (np.log(stdSeq) - np.log(ckH)) / kp)
            g1 = a1 * bH_2 - aH_1 * bH_1
            g2 = a1 * aH_2 - aH_1 * a2
            return (g1 if q == 0 else (g1 - a1 * H**q)) / g2

        # Solving the fixed point
        H = super().FixedPointSolver(
                LSSDIterFun, 0.5,
                n=n, p=p, q=q,
                kscale=kscale,
                stdSeq=stdSeq
            )

        return H

    def EstHurstLSV(self, ts: np.ndarray, max_scale: int, p=6, q=0) -> float:
        """
        Least Squares based on Variance. The fitting error is constructed by
        sample variances.

        Parameters
        ----------
        ts        : Time series.
        max_scale : Maximum aggregation scale(>= length / 10).
        p         : Parameter used to determine the weights.
        q         : Parameter used to determine the penalty factor.

        Returns
        -------
        The Hurst exponent of time series ts.

        References
        ----------
        [1] Tyralis H, Koutsoyiannis D (2011) Simultaneous estimation of the
            parameters of the Hurst-Kolmogorov stochastic process. Stochastic
            Environmental Research & Risk Assessment 25(1):21–33.
            doi:10.1007/s004770100408x.

        Written by z.q.feng (2023.06.07).
        """
        n = len(ts)
        # This maximum value was chosen so that VarSeq can be estimated
        # from at least 10 data values.
        max_scale = min(max_scale, n // 10 + 1)
        varSeq = np.linspace(1, max_scale, max_scale)
        # {1, 2, ..., max_scale}
        kscale = np.linspace(1, max_scale, max_scale).astype(int)
        # Variance of each sample
        for scale in kscale:
            sample = np.sum(
                np.reshape(ts[n % scale:], [n // scale, scale]),
                axis=1
            )
            varSeq[scale - 1] = np.var(sample, ddof=1)

        def LSV_target(H: float, **kwargs) -> float:
            """
            Target function to minimize in LSV method.
            """
            n = kwargs["n"]
            p = kwargs["p"]
            q = kwargs["q"]
            kscale = kwargs["kscale"]
            varSeq = kwargs["varSeq"]
            f = n / kscale
            kp = kscale**p
            # Left side of eq.22 in Ref[1]
            d1 = np.sum(varSeq**2 / kp)
            # eq.17 in Ref[1]
            ckH = (f - f**(2 * H - 1)) / (f - 1)
            # eq.20 in Ref[1]
            a1H = np.sum((ckH**2 * kscale**(4 * H)) / kp)
            a2H = np.sum((ckH * kscale**(2 * H) * varSeq) / kp)
            r = d1 - a2H**2 / (a1H if a1H > 1e-8 else 1e-8)
            # In Ref[1], to avoid values of sigma tending to infinity
            # when H->1, penalty factor [H^(q+1)]/(q+1) for a high q(near 50)
            # is added to target function, see eq.24.
            return r if q == 0 else r + H**(q + 1) / (q + 1)

        # TODO: Choose an appropriate optimize algorithm.
        H = super().LocalMin(
                LSV_target,
                [0.001, 0.999],
                n=n, p=p, q=q,
                kscale=kscale,
                varSeq=varSeq
            )

        return H

    def __GHESample(self, ts, tau):
        """
        Generates sample sequence as |X(t+τ)-X(t)| for t in {0, 1, ..., N-τ-1}.
        """
        return np.abs(ts[tau:] - ts[:-tau])

    def EstHurstGHE(self, ts, q=2, method="L2"):
        """
        Generalized Hurst Exponent. Estimated the Hurst exponent by using the
        qth-order moments of the distribution of the increments.

        Parameters
        ----------
        ts : Time sequence.
        q  : Moments order.

        Returns
        -------
        Hurst exponent of the time sequence

        References
        ----------
        [1] Albert-László Barabási and Tamás Vicsek. Multifractality of self-
            affine fractals. Physical review A, 44(4):2730, 1991.
        [2] Tiziana Di Matteo, Tomaso Aste, and Michel M Dacorogna. Scaling
            behaviors in diﬀerently developed markets. Physica A: statistical
            mechanics and its applications, 324(1-2):183–188, 2003.
        [3] A Gómez-Águila, JE Trinidad-Segovia, and MA Sánchez-Granero.
            Improvement in hurst exponent estimation and its application to
            ﬁnancial markets. Financial Innovation, 8(1):1–21, 2022.
        """
        Y = np.cumsum(ts - np.mean(ts))
        K, Tau = [], [i for i in range(1, 11)]
        for tau in Tau:
            # see Ref[3] for details.
            K.append(np.mean(self.__GHESample(Y, tau)**q))
        # k_q(τ) \propto τ^{qH(q)} where H(q) is the generalized Hurst exponent
        slope = self.__FitCurve(Tau, K, method=method)
        return slope / q

    def EstHurstKS(self, ts):
        """
        Kolmogorov-Smirnov Method. In most cases, existing methods use the
        scaling behavior (a power law) of certain elements of the process,
        this method take expected values of the equality in distribution to
        estimate the Hurst exponent. However, equality in distribution is a
        stronger concept than that in expected values.

        Parameters
        ----------
        ts : Time sequence.

        Returns
        -------
        Hurst exponent of the time sequence.

        References
        ----------
        [1] A Gómez-Águila, JE Trinidad-Segovia, and MA Sánchez-Granero.
            Improvement in hurst exponent estimation and its application to
            ﬁnancial markets. Financial Innovation, 8(1):1–21, 2022.
        [2] JL Hodges Jr. The signiﬁcance probability of the smirnov two-sample
            test. Arkiv för matematik, 3(5):469–486, 1958.
        """
        N = len(ts)
        Y = np.cumsum(ts - np.mean(ts))
        scaling_range = [2**i for i in range(int(np.log2(N)) - 2)]
        t0 = self.__GHESample(Y, scaling_range[0])

        def KS_target(H, t0):
            # see Ref[2] for details of Kolmogorov-Smirnov test.
            return np.sum(
                [stats.ks_2samp(t0, self.__GHESample(Y, tau)/tau**H).statistic
                 for tau in scaling_range[1:]]
            )

        H = super().LocalMin(KS_target, [0.001, 0.999], t0=t0)
        return H

    def EstHurstTTA(self, ts, max_scale=11, method="L2"):
        """
        Triangle Total Areas Method. Hurst exponent is the possibility of high
        or low values occurrence in time sequences. Based on this fact,
        triangle area of three samples in time sequences can be an important
        parameter for evaluation of series values. If three samples in time
        sequences have the same values, then the area of the mated triangle of
        this three samples has the lowest value.

        Parameters
        ----------
        ts        : Time sequence.
        max_scale : Maximum interval time for each triangle.

        Returns
        -------
        Hurst exponent of the time sequence.

        References
        ----------
        [1] Hamze Lotfalinezhad and Ali Maleki. TTA, a new approach to
            estimate hurst exponent with less estimation error and computa-
            tional time. Physica A: statistical mechanics and its applications,
            553:124093, 2020.
        [2] A Gómez-Águila and MA Sánchez-Granero. A theoretical framework
            for the tta algorithm. Physica A: Statistical Mechanics and its
            Applications, 582:126288, 2021.
        """
        Y = np.cumsum(ts - np.mean(ts))
        # τ is the interval points between first, middle and last point
        ST, Tau = [], [i for i in range(1, max_scale)]
        for tau in Tau:
            area = []
            # Area is the half of absulote determinant of matrix below:
            # | i,           ts[i],           1 |
            # | i + tau,     ts[i + tau],     1 |
            # | i + 2 * tau, ts[i + 2 * tau], 1 |
            for i in range(0, len(Y) - 2 * tau, 2 * tau):
                area.append(
                    abs(Y[i + 2 * tau] - 2 * Y[i + tau] + Y[i])
                )
            ST.append(0.5 * tau * np.sum(area))
        slope = self.__FitCurve(Tau, ST, method=method)
        return slope

    def EstHurstTA(self, ts, q=2, method="L2"):
        """
        Triangle Areas Method. the modiﬁcation is just to consider the
        distribution of the area of the triangles, instead of the distribution
        of the sum of the areas of all the triangles.

        Parameters
        ----------
        ts : Time sequence.
        q  : Moments order.

        Returns
        -------
        Hurst exponent of the time sequence.

        References
        ----------
        [1] A Gómez-Águila and MA Sánchez-Granero. A theoretical framework
            for the tta algorithm. Physica A: Statistical Mechanics and its
            Applications, 582:126288, 2021.
        """
        Y = np.cumsum(ts - np.mean(ts))
        ST, Tau = [], [2**i for i in range(int(np.log2(len(ts))) - 1)]

        # TODO: Find the correct distribution.
        for tau in Tau:
            area = (0.5 * tau * abs(Y[2 * tau] - 2 * Y[tau] + Y[0]))**q
            ST.append(area)

        slope = self.__FitCurve(Tau, ST, method=method)

        return slope / q - 1


if __name__ == "__main__":
    h = []
    N, H, m = 10000, 0.8, 15
    HSolver = HurstIndexSolver()
    for i in range(m):
        ts = fgn(N, H)
        # ts = np.loadtxt("reactionData.txt")[6, 1:]
        # ts = np.random.randn(1, N)[0]
        # ts = np.load("test.npy")
        h.append(HSolver.EstHurstLSV(ts, 10))
    print(np.mean(h))
