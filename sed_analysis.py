"""
sed_analysis.py - WANA spectral energy distribution analysis program

Analyze continuum images, compute flux densities, plot
flux vs frequency and flux vs uv-distance, fit SED

Copyright(C) 2019 by
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
Trey V. Wenger April 2019 - V1.0
"""

import os
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm

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

def main(field,peak_regions,full_regions,spws,label,
         taper=False,imsmooth=False,
         freqmin=7800,freqmax=10100,
         skip_plot=False):
    """
    Measure continuum flux density and rms for each source in the
    continuum MFS images. Plot and fit a power law to SED. Also plot
    and fit a curve to the flux vs UV distance in each image within
    the watershed region.

    Figures are saved as:
    - <label>.peak.cont_sed.pdf
      Continuum flux vs frequency with error bars
    - <label>.full.cont_sed.pdf
      Continuum flux vs frequency with error bars
    - <label>.<spw>.vis.pdf
      Visibility amplitude vs uv distance
    File is saved as:
    - <label>.cont_sed.txt
      Continuum fluxes, etc.

    Inputs:
      field :: string
        The field name
      peak_regions :: list of strings
        If only one element, the peak region file used to analyze each
        spw. Otherwise, the peak region file to use for each spw.
      full_regions :: list of strings
        If only one element, the watershed region file used to analyze each
        spw. Otherwise, the watershed region file to use for each spw.
      spws :: string
        comma-separated string of spws to analyze
      label :: string
        The first part of the output figure and filenames
      taper :: boolean
        if True, use uv-tapered images
      imsmooth :: boolean
        if True, use imsmooth images

    Returns: Nothing
    """
    #
    # If supplied only one region, expand to all spws
    #
    spws = ['spw{0}'.format(spw) if spw != 'cont' else 'cont'
            for spw in spws.split(',')]
    if len(peak_regions) == 1:
        peak_regions = [peak_regions[0] for spw in spws]
    if len(full_regions) == 1:
        full_regions = [full_regions[0] for spw in spws]
    #
    # Set-up file
    #
    outfile = '{0}.cont_sed.txt'.format(label)
    with open(outfile,'w') as f:
        # 0    1        2           3          4          5           6          7           8          9
        # spw  fluxtype frequency   cont       rms        area_arcsec area_pixel beam_arcsec beam_pixel ps_check
        # #             MHz         mJy/(beam) mJy/(beam) arcsec2     pixels     arcsec2     pixels     
        # cont peak     9494.152594 12345.6789 123.4567   12345.67    123456.78  12345.67    123456.78  100.00
        # 1    total    9494.152594 12345.6789 123.4567   12345.67    123456.78  12345.67    123456.78  100.00
        # 1234 12345678 12345678902 1234567890 1234567890 12345678901 1234567890 12345678901 1234567890 12345678
        #
        headerfmt = '{0:4} {1:8} {2:12} {3:10} {4:10} {5:11} {6:10} {7:11} {8:10} {9:8}\n'
        rowfmt = '{0:4} {1:8} {2:12.6f} {3:10.4f} {4:10.4f} {5:11.2f} {6:10.2f} {7:11.2f} {8:10.2f} {9:8.2f}\n'
        f.write(headerfmt.format('spw','fluxtype','frequency','cont','rms','area_arcsec','area_pixel','beam_arcsec','beam_pixel','ps_check'))
        fluxunit='mJy/(beam)'
        f.write(headerfmt.format('#','','MHz',fluxunit,fluxunit,'arcsec2','pixels','arcsec2','pixels',''))
        #
        # Read data for each spw
        #
        for spw,peak_region,full_region in zip(spws,peak_regions,full_regions):
            #
            # Read image
            #
            image = '{0}.{1}.mfs.clean'.format(field,spw)
            if taper: image += '.uvtaper'
            image += '.pbcor'
            if imsmooth: image += '.imsmooth'
            image += '.image.fits'
            if not os.path.exists(image):
                print("{0} not found.".format(image))
                continue
            image_hdu = fits.open(image)[0]
            image_wcs = WCS(image_hdu.header)
            wcs_celest = image_wcs.sub(['celestial'])
            frequency = image_hdu.header['CRVAL3']/1.e6 # MHz
            #
            # Calculate beam area and pixel size
            #
            pixel_size = 3600.**2. * np.abs(image_hdu.header['CDELT1'] * image_hdu.header['CDELT2']) # arcsec^2
            beam_arcsec = 3600.**2. * np.pi*image_hdu.header['BMIN']*image_hdu.header['BMAJ']/(4.*np.log(2.)) # arcsec^2
            beam_pixel = beam_arcsec / pixel_size
            #
            # Read residual image
            #
            residualimage = '{0}.{1}.mfs.clean'.format(field,spw)
            if taper: residualimage += '.uvtaper'
            if imsmooth: residualimage += '.imsmooth'
            residualimage += '.residual.fits'
            residual_hdu = fits.open(residualimage)[0]
            image_rms = 1.4825*np.nanmean(np.abs(residual_hdu.data[0,0]-np.nanmean(residual_hdu.data[0,0])))
            #
            # PB image, and PB-corrected rms
            #
            pbimage = '{0}.{1}.mfs.pb.fits'.format(field,spw)
            pb_hdu = fits.open(pbimage)[0]
            image_rms = image_rms/pb_hdu.data[0,0]
            #
            # Read region files, extract data from region center pixel
            #
            if not os.path.exists(peak_region):
                peak_cont = np.nan
                peak_rms = np.nan
                peak_area_pixel = np.nan
                peak_area_arcsec = np.nan
                peak_ps_check = np.nan
            else:
                with open(peak_region,'r') as freg:
                    freg.readline()
                    data = freg.readline()
                    splt = data.split(' ')
                    RA = splt[1].replace('[[','').replace(',','')
                    RA_h, RA_m, RA_s = RA.split(':')
                    RA = '{0}h{1}m{2}s'.format(RA_h,RA_m,RA_s)
                    dec = splt[2].replace('],','')
                    dec_d, dec_m, dec_s, dec_ss = dec.split('.')
                    dec = '{0}d{1}m{2}.{3}s'.format(dec_d,dec_m,dec_s,dec_ss)
                    coord = SkyCoord(RA,dec,frame='fk5')
                #
                # Get pixel location of coord, extract flux
                # N.B. Need to flip vertices to match WCS
                #
                peak_area_pixel = 1.
                peak_area_arcsec = pixel_size
                pix = wcs_celest.wcs_world2pix(coord.ra.deg,coord.dec.deg,1)
                peak_cont = 1000.*image_hdu.data[0,0,int(pix[1]),int(pix[0])] # mJy/beam
                #
                # Compute primary beam corrected rms at peak location
                #
                peak_rms = 1000.*image_rms[int(pix[1]),int(pix[0])]
                peak_ps_check = np.nan
            # write row to table
            f.write(rowfmt.format(spw.replace('spw',''),'peak',frequency,peak_cont,peak_rms,peak_area_arcsec,peak_area_pixel,beam_arcsec,beam_pixel,peak_ps_check))
            #
            # Read region file, sum spectrum region
            #
            if not os.path.exists(full_region):
                full_cont = np.nan
                full_rms = np.nan
                full_area_pixel = np.nan
                full_area_arcsec = np.nan
                full_ps_check = np.nan
            else:
                region_mask = np.array(fits.open(full_region)[0].data[0,0],dtype=np.bool)
                full_area_pixel = np.sum(region_mask)
                full_area_arcsec = full_area_pixel * pixel_size
                #
                # Extract sub-image
                #
                sub_image = image_hdu.data[0,0] * region_mask
                sub_image[np.isnan(sub_image)] = 0.
                #
                # Extract flux, mJy/beam*pixels/beam_area_pixels
                #
                full_cont = 1000.*np.nansum(sub_image)/beam_pixel # mJy
                #
                # Compute primary beam corrected rms
                # Since noise is spatially correlated on beam scales
                # rms = mJy/beam*pixels / beam_area_pixels * (region_area/beam_area) 
                #
                full_rms = 1000.*np.sqrt(np.sum(image_rms[region_mask]**2.))/beam_pixel * (full_area_pixel/beam_pixel)# mJy
                #
                # Generate synthesized beam kernel 2D Gaussian
                #
                # pixel grid
                x_size, y_size = image_hdu.data[0,0].shape
                x_axis = np.arange(x_size)
                y_axis = np.arange(y_size)
                x_grid, y_grid = np.meshgrid(x_axis,y_axis,indexing='ij')
                x_grid = x_grid - x_size/2
                y_grid = y_grid - y_size/2
                # angular offset grid (deg)
                x_grid = -x_grid * image_hdu.header['CDELT1']
                y_grid = y_grid * image_hdu.header['CDELT2']
                # beam size (sigma) in deg
                # PA in radians, converted from north through east
                # to east through north
                bmin = image_hdu.header['bmin']/(2.*np.sqrt(2.*np.log(2.)))
                bmaj = image_hdu.header['bmaj']/(2.*np.sqrt(2.*np.log(2.)))
                bpa = -np.deg2rad(image_hdu.header['bpa'])+np.pi/2.
                # 2-D Gaussian parameters
                A = np.cos(bpa)**2./(2.*bmaj**2.) + np.sin(bpa)**2./(2.*bmin**2.)
                B = -np.sin(2.*bpa)/(4.*bmaj**2.) + np.sin(2.*bpa)/(4.*bmin**2.)
                C = np.sin(bpa)**2./(2.*bmaj**2.) + np.cos(bpa)**2./(2.*bmin**2.)
                beam_kernel = np.exp(-(A*x_grid**2. + 2.*B*x_grid*y_grid + C*y_grid**2.)).T
                #
                # FFT the image and beam to get visibilities
                # Divide to deconvolve
                #
                sub_vis = np.fft.fftshift(np.fft.fft2(sub_image))
                beam_vis = np.fft.fftshift(np.fft.fft2(beam_kernel))
                beam_vis[np.abs(beam_vis) < 1.e-3] = np.nan
                deconvolve_vis = sub_vis/beam_vis
                #
                # FFT the grid to get uv distnaces
                #
                u_axis = np.fft.fftshift(np.fft.fftfreq(len(x_axis), d=-np.deg2rad(image_hdu.header['CDELT1'])))
                v_axis = np.fft.fftshift(np.fft.fftfreq(len(y_axis), d=np.deg2rad(image_hdu.header['CDELT2'])))
                u_grid, v_grid = np.meshgrid(u_axis,v_axis,indexing='xy')
                uvwave = np.sqrt(u_grid**2. + v_grid**2.).T.flatten()
                vis = np.abs(deconvolve_vis.flatten())
                #
                # Mask large uvwaves
                #
                uvwave_max = 4.*np.log(2.)/(np.pi*np.deg2rad(image_hdu.header['bmaj']))
                vis[uvwave > uvwave_max] = np.nan
                uvwave[uvwave > uvwave_max] = np.nan
                #
                # Check if this is a point source by computing the
                # percent difference between first 1000 wavelengths
                # and 1000 wavelengths before uvwave_max
                #
                is_zero = (uvwave < 1000)
                vis_zero = np.nanmean(vis[is_zero])
                is_max = (uvwave < uvwave_max)*(uvwave > uvwave_max-1000)
                vis_max = np.nanmean(vis[is_max])
                full_ps_check = 100.*(vis_max-vis_zero)/vis_zero
                #
                # Plot and save figure
                #
                xmin = 0.
                xmax = uvwave_max
                ymin = -0.1*np.nanmax(vis)
                ymax = 1.1*np.nanmax(vis)
                if not skip_plot:
                    plt.ioff()
                    fig, ax = plt.subplots()
                    h = ax.hist2d(uvwave, vis,
                                bins=20, norm=LogNorm(),
                                range=[[xmin,xmax],[ymin,ymax]])
                    cb = fig.colorbar(h[3])
                    cb.set_label("Number of Pixels")
                    ax.set_xlabel(r"{\it uv} distance (wavelengths)")
                    ax.set_ylabel(r"Amplitude (Jy)")
                    title = '{0}.{1}'.format(label,spw)
                    ax.set_title(title)
                    fig.tight_layout()
                    fig.savefig('{0}.{1}.vis.pdf'.format(label,spw))
                    plt.close(fig)
                    plt.ion()
            # write row to table
            f.write(rowfmt.format(spw.replace('spw',''),'full',frequency,full_cont,full_rms,full_area_arcsec,full_area_pixel,beam_arcsec,beam_pixel,full_ps_check))
    #
    # Now we read the file and plot the SEDs
    #
    data = np.genfromtxt(outfile,dtype=None,names=True,encoding='UTF-8')
    is_peak_total = (data['spw'] == 'cont')*(data['fluxtype'] == 'peak')
    is_full_total = (data['spw'] == 'cont')*(data['fluxtype'] == 'full')
    is_peak = (data['spw'] != 'cont')*(data['fluxtype'] == 'peak')
    is_full = (data['spw'] != 'cont')*(data['fluxtype'] == 'full')
    #
    # curve fit range
    #
    xfit = np.linspace(freqmin,freqmax,100)
    #
    # Plot peak SED
    #
    if not skip_plot:
        plt.ioff()
        fig, (ax, res_ax) = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
        # catch bad data
        isnan = ((np.isnan(data[is_peak]['cont'])) | (data[is_peak]['cont'] <= 0.))
        xdata = data[is_peak]['frequency'][~isnan]
        ydata = data[is_peak]['cont'][~isnan]
        e_ydata = data[is_peak]['rms'][~isnan]
        # plot data
        ax.errorbar(xdata,ydata,yerr=e_ydata,fmt='o',color='k')
        # Fit curves if we have enough data
        if len(ydata) > 4:
            try:
                # fit power law
                fit,cov = curve_fit(line, np.log10(xdata), np.log10(ydata),
                                    sigma = e_ydata/(np.log(10.)*ydata), absolute_sigma=True,
                                    method='trf', loss='soft_l1')
                yfit = lambda x: 10.**line(np.log10(x),fit[0],fit[1])
                ax.plot(xfit,yfit(xfit),'k--',zorder=10,
                        label=r'$F_{{\nu,\rm C}} \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
                ax.legend(loc='upper right',fontsize=10)
                # plot residuals
                residuals = ydata - yfit(xdata)
                r2 = 1. - np.sum(residuals**2.)/np.sum((ydata-np.mean(ydata))**2.)
                res_ax.errorbar(xdata,residuals,yerr=e_ydata,fmt='o',color='k')
                res_ax.annotate(r"$R^2$ = {0:.1f}".format(r2),xy=(0.02,0.8),xycoords='axes fraction',fontsize=10)
            except:
                # fit failed
                pass
        # plot total continuum
        ax.plot([freqmin,freqmax],[data[is_peak_total]['cont'][0],data[is_peak_total]['cont'][0]],'k-')
        ax.fill_between([freqmin,freqmax],
                        [data[is_peak_total]['cont'][0]-data[is_peak_total]['rms'][0],data[is_peak_total]['cont'][0]-data[is_peak_total]['rms'][0]],
                        [data[is_peak_total]['cont'][0]+data[is_peak_total]['rms'][0],data[is_peak_total]['cont'][0]+data[is_peak_total]['rms'][0]],
                        color='k',alpha=0.5,edgecolor='none')
        # Set plot axes
        ax.set_xlim(freqmin,freqmax)
        ax.set_ylabel('Flux Density (mJy/beam)')
        title = '{0}.peak'.format(label)
        ax.set_title(title)
        # Set residual plot axes
        res_ax.axhline(0.,color='k',lw=1.5)
        res_ax.set_xlabel('Frequency (MHz)')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1)
        fig.savefig('{0}.peak.cont_sed.pdf'.format(label))
        plt.close(fig)
        plt.ion()
        #
        # Plot full SED
        #
        plt.ioff()
        fig, (ax, res_ax) = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
        # catch bad data
        isnan = ((np.isnan(data[is_full]['cont'])) | (data[is_full]['cont'] <= 0.))
        xdata = data[is_full]['frequency'][~isnan]
        ydata = data[is_full]['cont'][~isnan]
        e_ydata = data[is_full]['rms'][~isnan]
        # plot data
        ax.errorbar(xdata,ydata,yerr=e_ydata,fmt='o',color='k')
        # Fit curves if we have enough data
        if len(ydata) > 4:
            try:
                # fit power law
                fit,cov = curve_fit(line, np.log10(xdata), np.log10(ydata),
                                    sigma = e_ydata/(np.log(10.)*ydata), absolute_sigma=True,
                                    method='trf', loss='soft_l1')
                yfit = lambda x: 10.**line(np.log10(x),fit[0],fit[1])
                ax.plot(xfit,yfit(xfit),'k--',zorder=10,
                        label=r'$F_{{\nu,\rm C}} \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
                ax.legend(loc='upper right',fontsize=10)
                # plot residuals
                residuals = ydata - yfit(xdata)
                r2 = 1. - np.sum(residuals**2.)/np.sum((ydata-np.mean(ydata))**2.)
                res_ax.errorbar(xdata,residuals,yerr=e_ydata,fmt='o',color='k')
                res_ax.annotate(r"$R^2$ = {0:.1f}".format(r2),xy=(0.02,0.8),xycoords='axes fraction',fontsize=10)
            except:
                # fit failed
                pass
        # plot total continuum
        ax.plot([freqmin,freqmax],[data[is_full_total]['cont'][0],data[is_full_total]['cont'][0]],'k-')
        ax.fill_between([freqmin,freqmax],
                        [data[is_full_total]['cont'][0]-data[is_full_total]['rms'][0],data[is_full_total]['cont'][0]-data[is_full_total]['rms'][0]],
                        [data[is_full_total]['cont'][0]+data[is_full_total]['rms'][0],data[is_full_total]['cont'][0]+data[is_full_total]['rms'][0]],
                        color='k',alpha=0.5,edgecolor='none')
        # Set plot axes
        ax.set_xlim(freqmin,freqmax)
        ax.set_ylabel('Flux Density (mJy)')
        title = '{0}.full'.format(label)
        ax.set_title(title)
        # Set residual plot axes
        res_ax.axhline(0.,color='k',lw=1.5)
        res_ax.set_xlabel('Frequency (MHz)')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1)
        fig.savefig('{0}.full.cont_sed.pdf'.format(label))
        plt.close(fig)
        plt.ion()
    
