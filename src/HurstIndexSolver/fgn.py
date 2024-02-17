#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:46:45 2023

@author: zqfeng
"""

import warnings
import numpy as np


def autocovariance(H, k):
    """Autocovariance for fgn."""
    return 0.5 * (
                abs(k - 1) ** (2 * H) -
                2 * abs(k) ** (2 * H) +
                abs(k + 1) ** (2 * H)
            )


def fgn(N: int, H: float):
    """
    Generate a fgn realization using Davies-Harte method.

    Uses Davies and Harte method (exact method) from:
    Davies, Robert B., and D. S. Harte. "Tests for Hurst effect."
    Biometrika 74, no. 1 (1987): 95-101.

    Can fail if n is small and hurst close to 1. Falls back to Hosking
    method in that case. See:

    Wood, Andrew TA, and Grace Chan. "Simulation of stationary Gaussian
    processes in [0, 1] d." Journal of computational and graphical
    statistics 3, no. 4 (1994): 409-432.

    Sample the fractional Gaussian noise.
    """
    scale = (1.0 / N) ** H
    gn = np.random.normal(0.0, 1.0, N)

    # If hurst == 1/2 then just return Gaussian noise
    if H == 0.5:
        return gn * scale

    # Monte carlo consideration
    # Generate the first row of the circulant matrix
    row_component = [autocovariance(H, i) for i in range(1, N)]
    reverse_component = list(reversed(row_component))
    row = [autocovariance(H, 0)] + row_component + [0] + reverse_component

    # Get the eigenvalues of the circulant matrix
    # Discard the imaginary part (should all be zero in theory so
    # imaginary part will be very small)
    eigenvals = np.fft.fft(row).real

    # If any of the eigenvalues are negative, then the circulant matrix
    # is not positive definite, meaning we cannot use this method. This
    # occurs for situations where n is low and H is close to 1.
    # Fall back to using the Hosking method. See the following for a more
    # detailed explanation:
    #
    # Wood, Andrew TA, and Grace Chan. "Simulation of stationary Gaussian
    #     processes in [0, 1] d." Journal of computational and graphical
    #     statistics 3, no. 4 (1994): 409-432.
    if np.any([ev < 0 for ev in eigenvals]):
        warnings.warn(
            "Combination of increments n and Hurst value H "
            "invalid for Davies-Harte method. Reverting to Hosking method."
            " Occurs when n is small and Hurst is close to 1."
        )
        # Set method to hosking for future samples.
        # method = "hosking"
        # Don"t need to store eigenvals anymore.
        eigenvals = None

    # Generate second sequence of i.i.d. standard normals
    gn2 = np.random.normal(0.0, 1.0, N)

    # Resulting sequence from matrix multiplication of positive definite
    # sqrt(C) matrix with fgn sample can be simulated in this way.
    w = np.zeros(2 * N, dtype=complex)
    for i in range(2 * N):
        if i == 0:
            w[i] = np.sqrt(eigenvals[i] / (2 * N)) * gn[i]
        elif i < N:
            w[i] = np.sqrt(eigenvals[i] / (4 * N)) * (gn[i] + 1j * gn2[i])
        elif i == N:
            w[i] = np.sqrt(eigenvals[i] / (2 * N)) * gn2[0]
        else:
            w[i] = np.sqrt(eigenvals[i] / (4 * N)) *\
                (gn[2 * N - i] - 1j * gn2[2 * N - i])

    # Resulting z is fft of sequence w. Discard small imaginary part (z
    # should be real in theory).
    z = np.fft.fft(w)
    fgn = z[: N].real
    # Scale to interval [0, L]
    return fgn * scale
