"""
plot_stokes.py - WANA generate continuum stokes analysis figures

Generate the following figures:
- Q/I, U/I, V/I, and P/I vs. frequency
- Q/I, U/I, V/I, and P/I vs. pb_level
- region_Q_rms/map_Q_rms, and same for U and V vs. frequency

Copyright(C) 2018 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changelog:
Trey V. Wenger November 2018 - V1.0
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

__version__ = "1.0"

def main(stokesfile,label,title=None,freqmin=4000,freqmax=10000):
    """
    Read output file from stokes_analysis.py and plot the following:
    - <label>.stokes_frac_freq.pdf
      Polarization fractions vs frequency with error bars
    - <label>.stokes_frac_pb.pdf
      Polarization fractions vs pb_level with error bars
    - <label>.stokes_rms_freq.pdf
      Relative stokes rms vs. frequency
    
    Inputs:
      stokesfile :: string
        The file containing stokes_analysis.py output
      label :: string
        The first part of the figure filenames
      title :: string
        The figure title
      freqmin :: scalar
        minimum frequency axis (MHz)
      freqmax :: scalar
        maximum frequency axis (MHz)

    Returns: Nothing
    """
    #
    # Read data
    #
    data = np.genfromtxt(stokesfile,dtype=None,names=True,encoding='UTF-8')
    #
    # Get data, careful that some may be missing
    #
    all_spws = np.unique(data['spw'])
    freqs = np.array(
        [data['frequency'][data['spw'] == spw][0]
         for spw in all_spws])
    pb_levels = np.array(
        [data['pb_level'][data['spw'] == spw][0]
         for spw in all_spws])
    cont = np.zeros((4, len(all_spws)))*np.nan
    e_cont = np.zeros((4, len(all_spws)))*np.nan
    map_med = np.zeros((4, len(all_spws)))*np.nan
    map_rms = np.zeros((4, len(all_spws)))*np.nan
    reg_med = np.zeros((4, len(all_spws)))*np.nan
    reg_rms = np.zeros((4, len(all_spws)))*np.nan
    for i, stokes in enumerate('IQUV'):
        cont[i] = np.array(
            [data['cont'][(data['stokes'] == stokes)*(data['spw'] == spw)][0]
             if np.sum((data['stokes'] == stokes)*(data['spw'] == spw)) > 0
             else np.nan for spw in all_spws])
        e_cont[i] = np.array(
            [data['e_cont'][(data['stokes'] == stokes)*(data['spw'] == spw)][0]
             if np.sum((data['stokes'] == stokes)*(data['spw'] == spw)) > 0
             else np.nan for spw in all_spws])
        map_med[i] = np.array(
            [data['map_med'][(data['stokes'] == stokes)*(data['spw'] == spw)][0]
             if np.sum((data['stokes'] == stokes)*(data['spw'] == spw)) > 0
             else np.nan for spw in all_spws])
        map_rms[i] = np.array(
            [data['map_rms'][(data['stokes'] == stokes)*(data['spw'] == spw)][0]
             if np.sum((data['stokes'] == stokes)*(data['spw'] == spw)) > 0
             else np.nan for spw in all_spws])
        reg_med[i] = np.array(
            [data['reg_med'][(data['stokes'] == stokes)*(data['spw'] == spw)][0]
             if np.sum((data['stokes'] == stokes)*(data['spw'] == spw)) > 0
             else np.nan for spw in all_spws])
        reg_rms[i] = np.array(
            [data['reg_rms'][(data['stokes'] == stokes)*(data['spw'] == spw)][0]
             if np.sum((data['stokes'] == stokes)*(data['spw'] == spw)) > 0
             else np.nan for spw in all_spws])
    #
    # Compute ratios and uncertainties
    #
    IQUV_frac = np.array([cont[i]/cont[0] for i in range(4)])
    e_IQUV_frac = np.array([
        np.sqrt(IQUV_frac[i]**2. * ((e_cont[i]/cont[i])**2. + (e_cont[0]/cont[0])**2.))
        for i in range(4)])
    P_frac = np.sqrt(cont[1]**2. + cont[2]**2. - e_cont[3]**2.)/cont[0]
    e_P_frac = np.sqrt((cont[1]/P_frac/cont[0]**2.)**2.*e_cont[1]**2. +
                       (cont[2]/P_frac/cont[0]**2.)**2.*e_cont[2]**2. +
                       (P_frac/cont[0])**2.*e_cont[0]**2.)
    rms_frac = np.array([reg_rms[i]/map_rms[i] for i in range(4)])
    #
    # Plot continuum polarization vs. frequency
    #
    print("Plotting polarization vs. frequency...")
    plt.ioff()
    fig, ax = plt.subplots()
    for i, lab, marker in zip([1,2,3],
                                [r'$Q/I$',r'$U/I$',r'$V/I$'],
                                ['s', '^', 'v']):
        ax.errorbar(freqs, IQUV_frac[i], yerr=e_IQUV_frac[i],
                    fmt=marker, color='k', label=lab, alpha=0.7)
    ax.errorbar(freqs, P_frac, yerr=e_P_frac,
                fmt='o', color='k', label=r'$P/I$', alpha=0.7)
    #
    # Set plot axes
    #
    ax.set_xlim(freqmin,freqmax)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Polarization Fraction')
    ax.set_title(title)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('{0}.stokes_frac_freq.pdf'.format(label))
    plt.close(fig)
    print("Done!")
    #
    # Plot continuum polarization vs. frequency
    #
    print("Plotting polarization vs. pb_level...")
    plt.ioff()
    fig, ax = plt.subplots()
    for i, lab, marker in zip([1,2,3],
                                [r'$Q/I$',r'$U/I$',r'$V/I$'],
                                ['s', '^', 'v']):
        ax.errorbar(pb_levels, IQUV_frac[i], yerr=e_IQUV_frac[i],
                    fmt=marker, color='k', label=lab, alpha=0.7)
    ax.errorbar(pb_levels, P_frac, yerr=e_P_frac,
                fmt='o', color='k', label=r'$P/I$', alpha=0.7)
    #
    # Set plot axes
    #
    ax.set_xlim(0, 100)
    ax.set_xlabel('Primary Beam Response (%)')
    ax.set_ylabel('Polarization Fraction')
    ax.set_title(title)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('{0}.stokes_frac_pb.pdf'.format(label))
    plt.close(fig)
    print("Done!")
    #
    # Plot continuum polarization rms vs. frequency
    #
    print("Plotting polarization rms vs. frequency...")
    plt.ioff()
    fig, ax = plt.subplots()
    for i, lab, marker in zip([1,2,3],
                                [r'$Q$',r'$U$',r'$V$'],
                                ['s', '^', 'v']):
        ax.plot(freqs, rms_frac[i], linestyle='none',
                marker=marker, color='k', label=lab, alpha=0.7)
    #
    # Set plot axes
    #
    ax.set_xlim(freqmin,freqmax)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('RMS Fraction')
    ax.set_title(title)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('{0}.stokes_rms_freq.pdf'.format(label))
    plt.close(fig)
    print("Done!")
