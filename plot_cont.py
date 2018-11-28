"""
plot_cont.py - WANA generate continuum analysis figures

Generate the following figures:
- Continuum SED w/ spectral index fits
- Continuum size vs. frequency

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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

__version__ = "1.0"

def line(x,m,b):
    """
    Equation for a 1-D line
    y = m*x + b

    Inputs:
      x : scalar or 1-D array of scalars
          The x-positions at which to compute y
      m : scalar
          The slope
      b : scalar
          The offset

    Returns: y
      y : scalar or 1-D array of scalars
          Line evaluated at each x position
    """
    return m*x + b

def main(contfile,label,title=None,fluxtype='peak',
         freqmin=4000,freqmax=10000):
    """
    Read output file from cont_analysis.py and plot the following:
    - <label>.cont_sed.pdf
      Continuum flux vs frequency with error bars
    - <label>.size.pdf (if fluxtype='total')
      Continuum size vs frequency
    
    Inputs:
      contfile :: string
        The file containing cont_analysis.py output
      label :: string
        The first part of the figure filenames
      title :: string
        The figure title
      fluxtype :: string
        'total' if units are mJy, 'peak' if units are mJy/beam
      freqmin :: scalar
        minimum frequency axis (MHz)
      freqmax :: scalar
        maximum frequency axis (MHz)

    Returns: Nothing
    """
    #
    # Read data
    #
    data = np.genfromtxt(contfile,dtype=None,names=True)
    #
    # Determine which is total continuum data
    #
    is_total = data['spw'] == 'cont'
    total = data[is_total][0]
    #
    # curve fit range
    #
    xfit = np.linspace(freqmin,freqmax,100)
    #
    # Plot continuum SED and fit residuals
    #
    print("Plotting continuum SED...")
    plt.ioff()
    fig, (ax, res_ax) = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
    # catch bad data
    isnan = ((np.isnan(data['cont'])) | (data['cont'] <= 0.))
    xdata = data['frequency'][~is_total*~isnan]
    ydata = data['cont'][~is_total*~isnan]
    e_ydata = data['rms'][~is_total*~isnan]
    # plot data
    ax.errorbar(xdata,ydata,yerr=e_ydata,fmt='o',color='k')
    #
    # Fit curves if we have enough data
    #
    if len(ydata) > 4:
        try:
            #
            # fit power law
            #
            #fit,cov = np.polyfit(np.log10(xdata),np.log10(ydata),1,
            #                     w=np.log(10.)*ydata/e_ydata,cov=True)
            #yfit = lambda x: 10.**np.poly1d(fit)(np.log10(x))
            #ax.plot(xfit,yfit(xfit),'k--',zorder=10,
            #        label=r'$F_{{\nu,\rm C}} \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
            fit,cov = curve_fit(line, np.log10(xdata), np.log10(ydata),
                                sigma = np.log(10.)*ydata/e_ydata, absolute_sigma=True,
                                method='trf', loss='soft_l1')
            yfit = lambda x: 10.**line(np.log10(x),fit[0],fit[1])
            ax.plot(xfit,yfit(xfit),'k--',zorder=10,
                    label=r'$F_{{\nu,\rm C}} \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
            ax.legend(loc='upper right',fontsize=10)
            #
            # plot residuals
            #
            residuals = ydata - yfit(xdata)
            r2 = 1. - np.sum(residuals**2.)/np.sum((ydata-np.mean(ydata))**2.)
            res_ax.errorbar(xdata,residuals,yerr=e_ydata,fmt='o',color='k')
            res_ax.annotate(r"$R^2$ = {0:.1f}".format(r2),xy=(0.1,0.8),xycoords='axes fraction')
        except:
            # fit failed
            pass
    # plot total continuum
    ax.plot([freqmin,freqmax],[total['cont'],total['cont']],'k-')
    ax.fill_between([freqmin,freqmax],
                    [total['cont']-total['rms'],total['cont']-total['rms']],
                    [total['cont']+total['rms'],total['cont']+total['rms']],
                    color='k',alpha=0.5,edgecolor='none')
    #
    # Set plot axes
    #
    ax.set_xlim(freqmin,freqmax)
    ax.set_xlabel('Frequency (MHz)')
    if fluxtype == 'flux':
        ax.set_ylabel('Flux Density (mJy)')
    else:
        ax.set_ylabel('Flux Density (mJy/beam)')
    ax.set_title(title)
    #
    # Set residual plot axes
    #
    res_ax.axhline(0.,color='k',lw=1.5)
    res_ax.set_xlabel('Frequency (MHz)')
    if fluxtype == 'flux':
        ax.set_ylabel('Residual (mJy)')
    else:
        ax.set_ylabel('Residual (mJy/beam)')    
    fig.tight_layout()
    fig.savefig('{0}.cont_sed.pdf'.format(label))
    plt.close(fig)
    print("Done!")
    #
    # Plot size vs frequency
    #
    if fluxtype == 'total':
        print("Plotting continuum sizes...")
        plt.ioff()
        fig, ax = plt.subplots()
        # catch bad data
        isnan = np.isnan(data['area_arcsec'])
        xdata = data['frequency'][~is_total*~isnan]
        ydata = data['area_arcsec'][~is_total*~isnan]
        # plot data
        ax.plot(xdata,ydata,'ko')
        # plot beam size
        # ax.plot(data['frequency'][~is_total],data['beam_arcsec'][~is_total],'k--')
        # plot total continuum
        ax.plot([freqmin,freqmax],[total['area_arcsec'],total['area_arcsec']],'k-')
        ax.set_xlim(freqmin,freqmax)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel(r'Size (arcsec$^2$)')
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig('{0}.size.pdf'.format(label))
        plt.close(fig)
        print("Done!")
