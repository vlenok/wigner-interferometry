# sigproc - a part of the wigner-interferometry project.
# Copyright (C) 2026 Vladimir Lenok
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import iv
from math import log10, ceil
from scipy.signal import hilbert


# general parameters for the bandpass filter
# all in terms of small omega (oppenheim book)
channel_center          = 0.05   * np.pi # center of the bandpass "channel"
channel_width           = 1.0E-3 * np.pi
channel_tappering_width = 1.0E-2 * np.pi
delta_lp, delta_hp      = 0.001, 0.001 # magnitude tolerances from the tolerance scheme


def sinc(n, alpha, omega_central):
    tmp = [np.sin(omega_central * (n_ - alpha)) / (np.pi * (n_ - alpha)) if np.abs(n_ - alpha) > 1.0E-6 \
           else omega_central/np.pi \
           for n_ in n]
    return np.array(tmp)


def kaiser(n, alpha, beta):
    argument = (n - alpha) / alpha
    argument = argument*argument
    argument = 1. - argument
    argument = np.sqrt(argument)
    argument = beta*argument
    return iv(0, argument)/iv(0, beta)


def kaiser_beta(A):
    if A > 50.:
        return 0.1102*(A-8.7)
    elif A <= 50. and A >= 21:
        return 0.5842*((A-21.)**0.4) + 0.07886*(A-21.)
    else:
        return 0.


def kaiser_M(A, delta_omega):
    M = A - 8.0
    M = M / (2.285*delta_omega)
    return ceil(M)
    

def get_phase_shift_filter(time_delay):
    # Design requirements
    delta_sh       = 0.001
    delta_omega_sh = 0.001*np.pi
    omega_cut_sh   = 0.85*np.pi

    shift = time_delay

    # Design constant section
    A_sh  = -20.0*log10(delta_sh)
    beta_sh = kaiser_beta(A_sh)
    M_sh = kaiser_M(A_sh, delta_omega_sh)

    alpha_sh = 0.5*M_sh

    n_sh = np.linspace(0., M_sh, int(M_sh + 1))
    w_sh = kaiser(n_sh, 0.5*M_sh, beta_sh)
    init_band_sh = sinc(n_sh, alpha_sh - shift, omega_cut_sh)
    filt = w_sh*init_band_sh
    return filt


def get_lp_filter(omega_cut):
    # Design requirements
    delta       = 0.001
    delta_omega = 0.001*np.pi

    # Design constant section
    A  = -20.0*log10(delta)
    beta = kaiser_beta(A)
    M = kaiser_M(A, delta_omega)

    alpha = 0.5*M

    n = np.linspace(0., M, int(M + 1))
    w = kaiser(n, 0.5*M, beta)
    init_band = sinc(n, alpha, omega_cut)
    filt = w*init_band
    return filt


def dwv(signal1, signal2):
    N = signal1.shape[0] - 1
    signal2_conj    = np.conjugate(signal2)
    doubled_signal1 = np.stack((signal1, signal1)).T.flatten()
    doubled_signal2 = np.stack((signal2_conj, signal2_conj)).T.flatten()
    M = signal1.shape[0]//2

    tmp1 = [doubled_signal1[0: ii*4 + 1] * np.flip(doubled_signal2[0: ii*4 + 1]) for ii in range(M)]
    tmp1 = [item[:-1] for item in tmp1]
    tmp1 = [np.pad(item, int(0.5*(2*N - item.shape[0])), 'constant', constant_values=0.) for item in tmp1]
    tmp1 = [np.roll(item, int(0.5*item.shape[0])) for item in tmp1]

    tmp2 = [doubled_signal1[3 + ii*4: (M-1)*4 + 2] * np.flip(doubled_signal2[3 + ii*4: (M-1)*4 + 2]) for ii in range(M)]
    tmp2 = [item[:-1] for item in tmp2]
    tmp2 = [np.pad(item, int(0.5*(2*N - item.shape[0])), 'constant', constant_values=0.) for item in tmp2]
    tmp2 = [np.roll(item, int(0.5*item.shape[0])) for item in tmp2]

    tmp_main = tmp1 + tmp2
    tmp_main = np.array(tmp_main)

    return np.fft.fft(tmp_main, axis=1)
