#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wes June 21 16:44:17 2023

@author: zqfeng
"""

import sys
import numpy as np
import scipy.optimize as op


class AddMethods(object):

    def __init__(self):
        None

    def Divisors(self, N: int, minimal=20) -> list:
        D = []
        for i in range(minimal, N // minimal + 1):
            if N % i == 0:
                D.append(i)
        return D

    def findOptN(self, N: int, minimal=20) -> int:
        """
        Find such a natural number OptN that possesses the largest number of
        divisors among all natural numbers in the interval [0.99*N, N]
        """
        N0 = int(0.99 * N)
        # The best length is the one which have more divisiors
        Dcount = [len(self.Divisors(i, minimal)) for i in range(N0, N + 1)]
        OptN = N0 + Dcount.index(max(Dcount))
        return OptN

    def OLE_linprog(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''
        Ax = b (Given A & b, try to derive x)

        Parameters
        ----------
        A : matrix like. With shape m x n.
        b : array like. With shape n x 1.

        Returns
        -------
        x : Minimal L1 norm solution of the system of equations.

        Reference
        ---------
        YAO Jian-kang. An Algorithm for Minimizing l1-Norm to Overdetermined
        Linear Eguations[J]. JIANGXI SCE7NICE, 2007, 25(1): 1-4.
        (Available at:
        http://d.g.wanfangdata.com.cn/Periodical_jxkx200701002.aspx)

        冯志强,张鸿燕. 基于残差向量l_1范数最小化与基追踪的多元线性模型参数估计
        方法[J]. 海南师范大学学报(自然科学版),2022,35(03):250-259+267.
        (Available at:
         http://hsxblk.hainnu.edu.cn/ch/reader/view_abstract.aspx?file_no=20220303)

        Version: 1.0 writen by z.q.feng @2022.03.13
        '''
        A, b = np.array(A), np.array(b)
        if np.size(A, 0) < np.size(A, 1):
            raise ValueError('Matrix A rows must greater than columns!')
        m, n = A.shape
        # Trans A into two matrix(n x n and (m - n) x n)
        A1, A2 = A[:n, :], A[n:, :]
        if np.linalg.matrix_rank(A) >= n:
            # inverse of A1
            A1_ = np.linalg.pinv(A1)
        else:
            # Generalized inverse of A1
            A1_ = np.linalg.pinv(A1)
        # c_ij = A2 * A1_
        c = np.dot(A2, A1_)
        # r(n+1:m) = A2*inv(A1)*r(1:n) + d
        d = np.dot(c, b[:n]) - b[n:]
        # Basic-Pursuit, target function = sum(u, v)
        t = np.ones([2 * m, 1])
        # Aeq_ = [c I(m-n)]
        Aeq_ = np.hstack([-c, np.eye(m - n, m - n)])
        # Aeq[u v] = Aeq_ * (u - v)
        Aeq = np.hstack([Aeq_, -Aeq_])
        # u, v > 0
        bounds = [(0, None) for i in range(2 * m)]
        # r0 = [u; v]
        r0 = op.linprog(t, A_eq=Aeq, b_eq=d, bounds=bounds,
                        method='revised simplex')['x']
        # Minimal L1-norm residual vector, r = u - v
        r = np.array([r0[:m] - r0[m:]])
        # Solving compatible linear equations Ax = b + r
        # Generalized inverse solution
        x = np.linalg.pinv(A).dot(b + r.T)
        return x

    def FixedPointSolver(self, fun, x0, eps=1e-6, **kwargs):
        """
        Solving the fixed points.
        """
        k, k_max = 0, 10000
        x_guess, dist = x0, 1
        while dist > eps and k < k_max:
            x_improved = fun(x_guess, **kwargs)
            dist = abs(x_improved - x_guess)
            x_guess = x_improved
            k += 1
        return x_guess

    def LocalMin(self, fun, interval: list, **kwargs):
        """
            The method used is a combination of  golden  section  search  and
        successive parabolic interpolation.  convergence is never much slower
        than  that  for  a  Fibonacci search.  If fun has a continuous second
        derivative which is positive at the minimum (which is not  at  ax  or
        bx),  then  convergence  is  superlinear, and usually of the order of
        about  1.324....
            The function fun is never evaluated at two points closer together
        than  eps*abs(fmin)+(tol/3), where eps is  approximately  the  square
        root  of  the  relative  machine  precision.   if  fun  is a unimodal
        function and the computed values of  fun  are  always  unimodal  when
        separated  by  at least  eps*abs(x)+(tol/3), then  fmin  approximates
        the abcissa of the global minimum of fun on the interval  ax,bx  with
        an error less than  3*eps*abs(fmin)+tol.  if  fun  is  not  unimodal,
        then fmin may approximate a local, but perhaps non-global, minimum to
        the same accuracy.
            This function subprogram is a slightly modified  version  of  the
        python3 procedure  localmin  given in Ref[1] page79.

        Parameters
        ----------
        fun      : Abcissa approximating the point where fun attains a minimum.
        interval : Iterative interval of target minimum point.

        Returns
        -------
        The Hurst exponent of time series ts.

        References
        ----------
        [1] Richard Brent, Algorithms for Minimization without Derivatives,
            Prentice-Hall, Inc. (1973).

        Written by z.q.feng (2023.06.07).
        """
        a, b = min(interval), max(interval)
        # c is the squared inverse of the golden ratio
        c, d, e = (3 - 5**0.5) / 2, 0, 0
        # eps is approximately the square root of relative machine precision.
        tol = sys.float_info.epsilon**0.25
        eps = tol**2
        # the smallest 1.000... > 1 : tol1 = 1 + eps**2
        v = w = x = a + c * (b - a)
        fv = fw = fx = fun(x, **kwargs)
        # main loop starts here
        while True:
            m = (a + b) / 2
            tol1 = eps * abs(x) + tol / 3
            tol2 = tol1 * 2
            # check stopping criterion
            if abs(x - m) <= tol2 - (b - a) / 2:
                break
            p = q = r = 0
            # fit parabola
            if abs(e) > tol1:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = (q - r) * 2
                if q > 0:
                    p *= -1
                else:
                    q *= -1
                r, e = e, d
            if abs(p) >= abs(0.5 * q * r) or p <= q * (a-x) or p >= q * (b-x):
                # a golden-section step
                e = (b if x < m else a) - x
                d = c * e
            else:
                # a parabolic-interpolation step
                d = p / q
                u = x + d
                # f must not be evaluated too close to ax or bx
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if x < m else -tol1
            # f must not be evaluated too close to x
            if abs(d) >= tol1:
                u = x + d
            elif d > 0:
                u = x + tol1
            else:
                u = x - tol1
            fu = fun(u, **kwargs)
            # update  a, b, v, w, and x
            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v, fv, w, fw = w, fw, u, fu
                elif fu <= fv or v == x or v == w:
                    v, fv = u, fu
        # end of main loop
        return x


if __name__ == "__main__":
    N, w_min = 1000,15
    AddMethod = AddMethods()
    OptN = AddMethod.findOptN(N, minimal=w_min)
    print(OptN)
    print(AddMethod.Divisors(OptN, minimal=w_min))
