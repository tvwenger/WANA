"""
plot_line.py - WANA generate RRL analysis figures

Generate the following figures:
- RRL SED w/ spectral index fits
- RRL flux vs. frequency
- RRL FWHM vs. frequency
- line-to-continuum ratio vs. frequency
- electron temperature vs. frequency

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

def main(specfile,label,title=None,fluxtype='peak',
         freqmin=4000,freqmax=10000):
    """
    Read output file from line_analysis and plot the following only
    for the (a) component:
    - <label>.line_sed.pdf
      Line flux vs frequency with error bars
    - <label>.fwhm.pdf
      Line FWHM vs frequency with error bars
    - <label>.line2cont.pdf
      Line-to-continuum ratio vs frequency with error bars
    - <label>.te.pdf
      Electron temperature vs frequency with error bars
    
    Inputs:
      specfile :: string
        The file containing line_analysis.py output
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
    data = np.genfromtxt(specfile,dtype=None,names=True)
    if len(data) == 0:
        return
    #
    # curve fit range
    #
    xfit = np.linspace(freqmin,freqmax,100)
    #
    # Ignore multiple components except (a)
    #
    multcomps = ['(b)','(c)','(d)','(e)']
    is_multcomp = np.array([lineid[-3:] in multcomps for lineid in data['lineid']])
    data = data[~is_multcomp]
    if len(data) == 0:
        return
    #
    # Determine which are Hnalpha lines and which are stacked
    #
    is_Hnalpha = np.array([lineid[0] == 'H' for lineid in data['lineid']])
    is_stacked = ~is_Hnalpha
    stacked = data[is_stacked][-1]
    #
    # Plot line flux vs frequency
    #
    print("Plotting line SED...")
    fig, ax = plt.subplots()
    # catch bad data
    isnan = np.isnan(data['line'])
    xdata = data['frequency'][is_Hnalpha*~isnan]
    ydata = data['line'][is_Hnalpha*~isnan]
    xlimit = data['frequency'][is_Hnalpha*isnan]
    ylimit = 3.*data['rms'][is_Hnalpha*isnan]
    e_ydata = data['e_line'][is_Hnalpha*~isnan]
    # plot data
    ax.errorbar(xdata,ydata,yerr=e_ydata,fmt='o',color='k')
    # plot limits
    ax.plot(xlimit,ylimit,linestyle='none',marker=r'$\downarrow$',
            markersize=20,color='k')
    #
    # Fit curves if we have enough data
    #
    if len(ydata) > 4:
        try:
            # fit line 
            fit,cov = np.polyfit(xdata,ydata,1,w=1./e_ydata,cov=True)
            ax.plot(xfit,np.poly1d(fit)(xfit),'k-',zorder=10,
                    label=r'$F_{{\nu,\rm L}} = ({0:.3f}\pm{1:.3f})\times10^{{-3}}\nu + ({2:.3f}\pm{3:.3f})$'.format(fit[0]*1.e3,np.sqrt(cov[0,0])*1.e3,fit[1],np.sqrt(cov[1,1])))
            ax.legend(loc='upper left',fontsize=10)
        except:
            # fit failed
            pass
    # plot stacked line
    ax.plot([freqmin,freqmax],[stacked['line'],stacked['line']],'k-')
    ax.fill_between([freqmin,freqmax],
                    [stacked['line']-stacked['e_line'],stacked['line']-stacked['e_line']],
                    [stacked['line']+stacked['e_line'],stacked['line']+stacked['e_line']],
                    color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(freqmin,freqmax)
    ax.set_xlabel('Frequency (MHz)')
    if fluxtype == 'peak':
        ax.set_ylabel('RRL Flux Density (mJy/beam)')
    else:
        ax.set_ylabel('RRL Flux Density (mJy)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig('{0}.line_sed.pdf'.format(label))
    plt.close(fig)
    print("Done!")
    #
    # Plot FWHM vs frequency
    #
    print("Plotting FWHM vs frequency...")
    fig, ax = plt.subplots()
    # catch bad data
    isnan = np.isnan(data['fwhm'])
    xdata = data['frequency'][is_Hnalpha*~isnan]
    ydata = data['fwhm'][is_Hnalpha*~isnan]
    e_ydata = data['e_fwhm'][is_Hnalpha*~isnan]
    # plot data
    ax.errorbar(xdata,ydata,yerr=e_ydata,fmt='o',color='k')
    #
    # Fit curves if we have enough data
    #
    if len(ydata) > 4:
        try:
            # fit line 
            fit,cov = np.polyfit(xdata,ydata,1,w=1./e_ydata,cov=True)
            ax.plot(xfit,np.poly1d(fit)(xfit),'k-',zorder=10,
                    label=r'$\Delta V = ({0:.3f}\pm{1:.3f})\times10^{{-3}}\nu + ({2:.3f}\pm{3:.3f})$'.format(fit[0]*1.e3,np.sqrt(cov[0,0])*1.e3,fit[1],np.sqrt(cov[1,1])))
            ax.legend(loc='upper left',fontsize=10)
        except:
            # fit failed
            pass
    # plot stacked line
    ax.plot([freqmin,freqmax],[stacked['fwhm'],stacked['fwhm']],'k-')
    ax.fill_between([freqmin,freqmax],
                    [stacked['fwhm']-stacked['e_fwhm'],stacked['fwhm']-stacked['e_fwhm']],
                    [stacked['fwhm']+stacked['e_fwhm'],stacked['fwhm']+stacked['e_fwhm']],
                    color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(freqmin,freqmax)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('RRL FWHM (km/s)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig('{0}.fwhm.pdf'.format(label))
    plt.close(fig)
    print("Done!")
    #
    # Plot line-to-continuum ratio vs frequency
    #
    print("Plotting line-to-continuum ratio...")
    fig, ax = plt.subplots()
    # catch bad data
    isnan = np.isnan(data['line2cont'])
    xdata = data['frequency'][is_Hnalpha*~isnan]
    ydata = data['line2cont'][is_Hnalpha*~isnan]
    xlimit = data['frequency'][is_Hnalpha*isnan]
    ylimit = 3.*data['rms'][is_Hnalpha*isnan]/data['cont'][is_Hnalpha*isnan]
    e_ydata = data['e_line2cont'][is_Hnalpha*~isnan]
    # plot data
    ax.errorbar(xdata,ydata,yerr=e_ydata,fmt='o',color='k')
    # plot limits
    ax.plot(xlimit,ylimit,linestyle='none',marker=r'$\downarrow$',
            markersize=20,color='k')
    #
    # Fit curves if we have enough data
    #
    if len(ydata) > 4:
        try:
            # fit line 
            fit,cov = np.polyfit(xdata,ydata,1,w=1./e_ydata,cov=True)
            ax.plot(xfit,np.poly1d(fit)(xfit),'k-',zorder=10,
                    label=r'$F_{{\nu,\rm L}}/F_{{\nu,\rm C}} = ({0:.3f}\pm{1:.3f})\times10^{{-3}}\nu + ({2:.3f}\pm{3:.3f})$'.format(fit[0]*1.e3,np.sqrt(cov[0,0])*1.e3,fit[1],np.sqrt(cov[1,1])))
            ax.legend(loc='upper left',fontsize=10)
        except:
            # fit failed
            pass
    # plot stacked line
    ax.plot([freqmin,freqmax],[stacked['line2cont'],stacked['line2cont']],'k-')
    ax.fill_between([freqmin,freqmax],
                    [stacked['line2cont']-stacked['e_line2cont'],stacked['line2cont']-stacked['e_line2cont']],
                    [stacked['line2cont']+stacked['e_line2cont'],stacked['line2cont']+stacked['e_line2cont']],
                    color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(freqmin,freqmax)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Line-to-Continuum Ratio')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig('{0}.line2cont.pdf'.format(label))
    plt.close(fig)
    print("Done!")
    #
    # Plot electron temperature vs frequency
    #
    print("Plotting electron temperature...")
    fig, ax = plt.subplots()
    # catch bad data
    isnan = np.isnan(data['elec_temp'])
    xdata = data['frequency'][is_Hnalpha*~isnan]
    ydata = data['elec_temp'][is_Hnalpha*~isnan]
    e_ydata = data['e_elec_temp'][is_Hnalpha*~isnan]
    # plot data
    ax.errorbar(xdata,ydata,yerr=e_ydata,fmt='o',color='k')
    #
    # Fit curves if we have enough data
    #
    if len(ydata) > 4:
        try:
            # fit line 
            fit,cov = np.polyfit(xdata,ydata,1,w=1./e_ydata,cov=True)
            ax.plot(xfit,np.poly1d(fit)(xfit),'k-',zorder=10,
                    label=r'$T_e = ({0:.3f}\pm{1:.3f})\times10^{{-3}}\nu + ({2:.3f}\pm{3:.3f})$'.format(fit[0]*1.e3,np.sqrt(cov[0,0])*1.e3,fit[1],np.sqrt(cov[1,1])))
            ax.legend(loc='upper left',fontsize=10)
        except:
            # fit failed
            pass
    # plot stacked line
    ax.plot([freqmin,freqmax],[stacked['elec_temp'],stacked['elec_temp']],'k-')
    ax.fill_between([freqmin,freqmax],
                    [stacked['elec_temp']-stacked['e_elec_temp'],stacked['elec_temp']-stacked['e_elec_temp']],
                    [stacked['elec_temp']+stacked['e_elec_temp'],stacked['elec_temp']+stacked['e_elec_temp']],
                    color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(freqmin,freqmax)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Electron Temperature (K)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig('{0}.te.pdf'.format(label))
    plt.close(fig)
    print("Done!")
    plt.ion()
