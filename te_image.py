"""
te_image.py - WANA generate electron temperature images

Generate electron temperature images in two ways: using moment 0 
images and by fitting RRLs in each image pixel

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

Trey V. Wenger September 2019 -V2.0
    Update for WISP V2.0 with stokes parameter support
    Add smoothing and re-gridding
"""

import os

import glob
import itertools
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

__version__ = "2.0"

def line_free(xdata, ydata):
    """
    Find the line-free regions of ydata by iteratively fitting
    a 3rd order polynomial baseline and removing outliers

    Inputs: xdata, ydata
      xdata :: 1-D array of scalars
        The x-axis data 
      ydata :: 1-D array of scalars
        The y-axis data
    
    Returns: xgood, ygood
      xgood :: 1-D array of scalars
        The x-data without outliers
      ygood :: 1-D array of scalars
        The y-data without outliers
    """
    outliers = np.isnan(ydata)
    while True:
        if np.sum(outliers) == len(xdata):
            return np.array([]), np.array([])
        pfit = np.polyfit(xdata[~outliers], ydata[~outliers], 3)
        yfit = np.poly1d(pfit)
        new_ydata = ydata - yfit(xdata)
        rms = 1.4826*np.median(np.abs(new_ydata[~outliers]-np.median(new_ydata[~outliers])))
        new_outliers = (np.abs(new_ydata) > 3.*rms) | np.isnan(ydata)
        if np.sum(new_outliers) <= np.sum(outliers):
            break
        outliers = new_outliers
    return xdata[~outliers], ydata[~outliers]

def gauss_guess(xdata, ydata):
    """
    Remove outliers, get Gaussian parameter estimates

    Inputs: xdata, ydata
      xdata :: 1-D array of scalars
        The x-axis data 
      ydata :: 1-D array of scalars
        The y-axis data
    
    Returns: start, peak, center, sigma, end
      start :: scalar
        The start of the line region in x-data units
      peak :: scalar
        The Gaussian peak estimate
      center :: scalar
        The Gaussian center estimate
      sigma :: scalar
        The Gaussian width estimate
      end :: scalar
        The end of the line region in x-data units
    """
    #
    # Get outliers to identify line region
    #
    missing = np.isnan(ydata)
    outliers = np.array([False]*len(ydata))
    while True:
        exclude = missing+outliers
        rms = 1.4826*np.median(np.abs(ydata[~exclude]-np.median(ydata[~exclude])))
        new_outliers = (np.abs(ydata) > 3.*rms)
        if np.sum(new_outliers) <= np.sum(outliers):
            break
        outliers = new_outliers
    #
    # Group outliers to find largest outlier group
    #
    line_x = np.array([])
    for val,ch in itertools.groupby(range(len(xdata)), lambda x: outliers[x]):
        if val:
            chs = np.array(list(ch))
            # skip if outliers are negative, too small, or smaller
            # than the current saved region
            if np.sum(ydata[chs]) < 0 or len(chs) < 4 or len(chs) < len(line_x):
                continue
            line_x = xdata[chs]
    # if line_x is empty, no guess
    if len(line_x) == 0:
        return None, None, None, None, None
    #
    # get line parameters guesses
    #
    start = np.min(line_x) - 10
    end = np.max(line_x) + 10
    center = np.mean(line_x)
    peak = np.nanmax(ydata)
    sigma = (end - center) / 2.
    return start, peak, center, sigma, end

def gaussian(xdata,*args):
    """
    Compute sum of multiple Gaussian functions.

    Inputs: xdata, *args
      xdata :: ndarray of scalars
        The x-values at which to compute the Gaussian functions
      args :: iterable of scalars
        a0,c0,s0,...,an,cn,sn
        where a0 = amplitude of first Gaussian
              c0 = center of first Gaussian
              s0 = sigma width of first Gaussian
               n = number of Gaussian components

    Returns:
      ydata :: ndarray of scalars
        Sum of Gaussian functions evaluated at xdata.
    """
    if len(args) % 3 != 0:
        raise ValueError("gaussian() arguments must be multiple of 3")
    amps = args[0::3]
    centers = args[1::3]
    sigmas = args[2::3]
    ydata = np.zeros(len(xdata))
    for a,c,s in zip(amps,centers,sigmas):
        ydata += a*np.exp(-(xdata-c)**2./(2.*s**2.))
    return ydata

def smooth_regrid(hdu, velocity):
    """
    Smooth and re-grid a 3-D data cube to a common velocity grid.
    Use sinc function for smoothing and re-gridding.

    Inputs:
      hdu :: astropy.fits.HDU
        HDU container of data
      velocity :: 1-D array of scalars
        The new velocity axis at which to interpolate velocities

    Returns:
      newhdu :: astropy.fits.HDU
        The new HDU container with the smoothed/re-gridded data
    """
    smogrid_res = velocity[1]-velocity[0]
    original_velocity = ((np.arange(hdu.header['NAXIS3']) - (hdu.header['CRPIX3']-1))*hdu.header['CDELT3'] +
                         hdu.header['CRVAL3'])/1000. # km/s
    original_res = hdu.header['CDELT3']/1000. # km/s
    if smogrid_res < original_res:
        raise ValueError("Cannot smooth to a finer resolution!")
    # construct sinc weights, and catch out of bounds
    sinc_wts = np.array([np.sinc((v-original_velocity)/smogrid_res)
                         if (original_velocity[0] < v < original_velocity[-1])
                         else np.zeros(len(original_velocity))*np.nan
                         for v in velocity])
    # normalize
    sinc_wts = (sinc_wts.T/np.sum(sinc_wts, axis=1)).T
    # convolve
    newdata = np.tensordot(sinc_wts, hdu.data[0], axes=([1], [0]))[np.newaxis]
    # update hdu
    newhdu = hdu.copy()
    newhdu.data = newdata
    newhdu.header['NAXIS3'] = len(velocity)
    newhdu.header['CUNIT3'] = 'm/s'
    newhdu.header['CRVAL3'] = velocity[0]*1000.
    newhdu.header['CDELT3'] = smogrid_res*1000.
    newhdu.header['CRPIX3'] = 1
    return newhdu

def process(field, spw, uvtaper=False, imsmooth=False,
            smogrid_velocity=None):
    """
    Compute the moment 0 maps and generate electron temperature maps.
    Fit lines in each pixel and generate electron temperature maps.

    Inputs:
      field :: string
        The field name
      spw :: string
        The spectral window to analyze
      uvtaper :: boolean
        If True, use the uv-tapered image
      imsmooth :: boolean
        If True, use the imsmoothed image
      smogrid_velocity :: 1-D array of scalars
        If not None, the channel cubes are re-gridded to this velocity
        axis.

      Returns: goodplots
        goodplots :: list of strings
          The plot figure filenames generated by this function
    """
    goodplots = []
    if spw != 'all':
        spw = 'spw{0}'.format(spw)
    linetype = 'clean'
    if uvtaper: linetype += '.uvtaper'
    linetype += '.pbcor'
    if imsmooth: linetype += '.imsmooth'
    chanlinetype = linetype
    #
    # Read MFS image and channel image
    #
    mfsimage = '{0}.{1}.I.mfs.{2}.image.fits'.format(field,spw,linetype)
    mfshdu = fits.open(mfsimage)
    chanimage = '{0}.{1}.I.channel.{2}.image.fits'.format(field,spw,linetype)
    chanhdu = fits.open(chanimage)
    #
    # Smooth/regrid velocity axis of channel image
    #
    if smogrid_velocity is not None:
        chanhdu[0] = smooth_regrid(chanhdu[0], smogrid_velocity)
        # save smoothed image
        chanlinetype += '.smogrid'
        chanimage = '{0}.{1}.I.channel.{2}.image.fits'.format(field,spw,chanlinetype)
        chanhdu.writeto(chanimage, overwrite=True, output_verify='fix+warn')
    #
    # Clip MFS and channel images using fullrgn masks
    #
    rgnspw = spw
    if spw == 'all':
        rgnspw = 'all'
    fullrgn = '*.notaper.{0}.fullrgn.fits'.format(rgnspw)
    if uvtaper: fullrgn = '*.uvtaper.{0}.fullrgn.fits'.format(rgnspw)
    if imsmooth: fullrgn = '*.imsmooth.{0}.fullrgn.fits'.format(rgnspw)
    if uvtaper and imsmooth: fullrgn = '*.uvtaper.imsmooth.{0}.fullrgn.fits'.format(rgnspw)
    masks = glob.glob(fullrgn)
    cliprgn = np.zeros(mfshdu[0].data.shape, dtype=np.bool)
    for mask in masks:
        fullrgnhdu = fits.open(mask)
        fullrgn_data = np.array(fullrgnhdu[0].data,dtype=np.bool)
        cliprgn = cliprgn | fullrgn_data
    # clip mfs
    clipmfs = mfshdu[0].data
    clipmfs *= cliprgn
    clipmfs[np.where(clipmfs == 0)] = np.nan
    # clip channels
    clipchan = chanhdu[0].data
    for clipchannel in clipchan[0]:
        clipchannel *= cliprgn[0,0]
    clipchannel[np.where(clipchannel == 0)] = np.nan
    #
    # Save clipped images
    #
    mfsclipimage = '{0}.{1}.I.mfs.{2}.image.clip.fits'.format(field,spw,linetype)
    mfshdu[0].data = clipmfs
    mfshdu.writeto(mfsclipimage,overwrite=True, output_verify='fix+warn')
    chanclipimage = '{0}.{1}.I.channel.{2}.image.clip.fits'.format(field,spw,chanlinetype)
    chanhdu[0].data = clipchan
    chanhdu.writeto(chanclipimage,overwrite=True, output_verify='fix+warn')
    #
    # Analyze image on pixel by pixel basis
    #
    # continuum subtracted cube
    contsub = np.empty(clipchan.shape)*np.nan
    # spectral rms
    specrms = np.empty(clipchan.shape[2:])*np.nan
    # median continuum flux
    contmedian = np.empty(clipchan.shape[2:])*np.nan
    # number of continuum channels used
    contchans = np.empty(clipchan.shape[2:])*np.nan
    # fitted line peak flux
    lineflux = np.empty(clipchan.shape[2:])*np.nan
    e_lineflux = np.empty(clipchan.shape[2:])*np.nan
    # fitted line fwhm
    linefwhm = np.empty(clipchan.shape[2:])*np.nan
    e_linefwhm = np.empty(clipchan.shape[2:])*np.nan
    # fittend line velocity
    linevlsr = np.empty(clipchan.shape[2:])*np.nan
    e_linevlsr = np.empty(clipchan.shape[2:])*np.nan
    # sum of line flux
    linesum = np.empty(clipchan.shape[2:])*np.nan
    # number of line channels used
    linechans = np.empty(clipchan.shape[2:])*np.nan
    # the channels and velocities
    all_chans = np.arange(clipchan.shape[1])
    velocities = ((all_chans+1-chanhdu[0].header['CRPIX3'])*chanhdu[0].header['CDELT3']+chanhdu[0].header['CRVAL3'])/1000.
    velocity_width = chanhdu[0].header['CDELT3']/1000.
    #
    # Loop over all un-masked pixels in image
    #
    foo,bar,xnotmask,ynotmask = np.where(~np.isnan(clipmfs))
    for x,y in zip(xnotmask,ynotmask):
        #
        # Find line-free regions
        #
        chan, flux = line_free(all_chans,clipchan[0,:,x,y])
        if len(chan) == 0:
            # all nan
            continue
        #
        # Fit and remove 3rd order polynomial background
        #
        pfit = np.polyfit(chan, flux, 3)
        yfit = np.poly1d(pfit)
        flux_contsub = clipchan[0,:,x,y]-yfit(all_chans)
        contsub[0, :, x, y] = flux_contsub
        #
        # Compute continuum flux and rms
        #
        contmedian[x, y] = np.median(flux)
        contchans[x, y] = len(chan)
        specrms[x, y] = 1.4826*(np.median(np.abs(flux-np.median(flux))))
        #
        # Estimate line parameters
        #
        start, peak, center, sigma, end = gauss_guess(velocities, flux_contsub)
        if start is None:
            # no obvious line
            continue
        #
        # Fit Gaussian
        #
        bounds_lower = [0, start, 0]
        bounds_upper = [np.inf, end, np.inf]
        bounds = (bounds_lower,bounds_upper)
        p0 = (peak, center, sigma)
        isnan = np.isnan(flux_contsub)
        try:
            popt,pcov = curve_fit(gaussian,velocities[~isnan],
                                  flux_contsub[~isnan],
                                  p0=p0,bounds=bounds)
        except:
            # fit failed
            continue
        #
        # Check that line parameters are sane
        # - center is not within FWHM of edge
        # - FWHM are not < 5 km/s
        # - FWHM are not > 150 km/s
        # - intensity and FWHM errors are not > 100%
        #
        if ((popt[1] - popt[2] < np.min(velocities)) or
            (popt[1] + popt[2] > np.max(velocities)) or
            (popt[2]*2.*np.sqrt(2.*np.log(2.)) < 5.) or
            (popt[2]*2.*np.sqrt(2.*np.log(2.)) > 150.) or
            (np.sqrt(pcov[0,0])/popt[0] > 1.0) or
            (np.sqrt(pcov[2,2])/popt[2] > 1.0)):
            continue
        #
        # Save fit and fit errors
        #
        lineflux[x,y] = popt[0]
        e_lineflux[x,y] = np.sqrt(pcov[0,0])
        linevlsr[x,y] = popt[1]
        e_linevlsr[x,y] = np.sqrt(pcov[1,1])
        linefwhm[x,y] = popt[2]*2.*np.sqrt(2.*np.log(2.))
        e_linefwhm[x,y] = np.sqrt(pcov[2,2])*2.*np.sqrt(2.*np.log(2.))
        #
        # Sum spectrum from -3sigma to 3sigma
        #
        startind = np.argmin(np.abs(velocities-(popt[1]-3*popt[2])))
        endind = np.argmin(np.abs(velocities-(popt[1]+3*popt[2])))
        linesum[x,y] = np.sum(flux_contsub[startind:endind])*velocity_width
        linechans[x,y] = endind-startind
    #
    # Save images
    #
    # continuum subtracted cube
    contsubimage = '{0}.{1}.I.channel.{2}.image.contsub.fits'.format(field,spw,chanlinetype)
    chanhdu[0].data = contsub
    chanhdu.writeto(contsubimage,overwrite=True, output_verify='fix+warn')
    # sum of line flux
    linesumimage = '{0}.{1}.I.channel.{2}.image.linesum.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = linesum
    mfshdu[0].header['BUNIT'] = 'Jy/beam*km/s'
    mfshdu.writeto(linesumimage, overwrite=True, output_verify='fix+warn')
    goodplots.append(('linesum',spw,linesumimage))
    # number of line channels used
    linechansimage = '{0}.{1}.I.channel.{2}.image.linechans.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = linechans
    mfshdu[0].header['BUNIT'] = 'Number'
    mfshdu.writeto(linechansimage, overwrite=True, output_verify='fix+warn')
    # spectral rms
    specrmsimage = '{0}.{1}.I.channel.{2}.image.specrms.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = specrms
    mfshdu.writeto(specrmsimage,overwrite=True, output_verify='fix+warn')
    goodplots.append(('specrms',spw,specrmsimage))
    # median continuum flux
    contmedianimage = '{0}.{1}.I.channel.{2}.image.contmedian.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = contmedian
    mfshdu.writeto(contmedianimage,overwrite=True, output_verify='fix+warn')
    # number of continuum channels used
    contchansimage = '{0}.{1}.I.channel.{2}.image.contchans.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = contchans
    mfshdu[0].header['BUNIT'] = 'Number'
    mfshdu.writeto(contchansimage, overwrite=True, output_verify='fix+warn')
    # fitted line peak flux
    linefluximage = '{0}.{1}.I.channel.{2}.image.lineflux.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = lineflux
    mfshdu[0].header['BUNIT'] = 'Jy/beam'
    mfshdu.writeto(linefluximage, overwrite=True, output_verify='fix+warn')
    e_linefluximage = '{0}.{1}.I.channel.{2}.image.e_lineflux.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = e_lineflux
    mfshdu.writeto(e_linefluximage, overwrite=True, output_verify='fix+warn')
    goodplots.append(('lineflux',spw,linefluximage,e_linefluximage))
    # fitted line fwhm
    linefwhmimage = '{0}.{1}.I.channel.{2}.image.linefwhm.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = linefwhm
    mfshdu[0].header['BUNIT'] = 'km/s'
    mfshdu.writeto(linefwhmimage, overwrite=True, output_verify='fix+warn')
    e_linefwhmimage = '{0}.{1}.I.channel.{2}.image.e_linefwhm.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = e_linefwhm
    mfshdu.writeto(e_linefwhmimage, overwrite=True, output_verify='fix+warn')
    goodplots.append(('linefwhm',spw,linefwhmimage,e_linefwhmimage))
    # fittend line velocity
    linevlsrimage = '{0}.{1}.I.channel.{2}.image.linevlsr.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = linevlsr
    mfshdu[0].header['BUNIT'] = 'km/s'
    mfshdu.writeto(linevlsrimage, overwrite=True, output_verify='fix+warn')
    e_linevlsrimage = '{0}.{1}.I.channel.{2}.image.e_linevlsr.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = e_linevlsr
    mfshdu.writeto(e_linevlsrimage, overwrite=True, output_verify='fix+warn')
    goodplots.append(('linevlsr',spw,linevlsrimage,e_linevlsrimage))
    #
    # Compute electron temperature image using summed line
    #
    freq = mfshdu[0].header['RESTFRQ']/1.e9
    tesum = (7103.3 * freq**1.1 * contmedian * (2. * linesum * np.sqrt(np.log(2.)/np.pi))**-1. * (1.08)**-1.)**0.87
    e_tesum = 0.87 * tesum * specrms * np.sqrt(1./(contmedian**2. * contchans) + linechans*velocity_width**2./linesum**2.)
    #
    # Save sum Te image
    #
    tesumimage = '{0}.{1}.I.channel.{2}.image.tesum.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = tesum
    mfshdu[0].header['BUNIT'] = 'K'
    mfshdu.writeto(tesumimage, overwrite=True, output_verify='fix+warn')
    e_tesumimage = '{0}.{1}.I.channel.{2}.image.e_tesum.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = e_tesum
    mfshdu.writeto(e_tesumimage, overwrite=True, output_verify='fix+warn')
    goodplots.append(('tesum',spw,tesumimage,e_tesumimage))
    #
    # Compute electron temperature using line fits
    #
    tefit = (7103.3 * freq**1.1 * contmedian * (lineflux * linefwhm)**-1. * (1.08)**-1.)**0.87
    e_tefit = 0.87 * tefit * np.sqrt((specrms/contmedian)**2.* 1./contchans + (e_lineflux/lineflux)**2. + (e_linefwhm/linefwhm)**2.)
    #
    # Save fit Te image
    #
    tefitimage = '{0}.{1}.I.channel.{2}.image.tefit.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = tefit
    mfshdu[0].header['BUNIT'] = 'K'
    mfshdu.writeto(tefitimage, overwrite=True, output_verify='fix+warn')
    e_tefitimage = '{0}.{1}.I.channel.{2}.image.e_tefit.fits'.format(field,spw,chanlinetype)
    mfshdu[0].data = e_tefit
    mfshdu.writeto(e_tefitimage, overwrite=True, output_verify='fix+warn')
    goodplots.append(('tefit',spw,tefitimage,e_tefitimage))
    return goodplots

def main(field,spws='',uvtaper=False,imsmooth=False,stack=False,
         smogrid_start=None, smogrid_res=None, smogrid_end=None):
    """
    For each line spectral window, generate:
    1. continuum-subtracted cube
    2. mean continuum image
    3. line moment 0 image
    4. electron temperature image
    5. electron temperature uncertainty image
    6. clipped electron temperature image

    Generate a PDF with the final images.

    Inputs:
      field :: string
        The field we are working on
      spws :: string
        Comma-separated spectral windows to image
      uvtaper :: boolean
        If True, use the uv-tapered images
      imsmooth :: boolean
        If True, use the imsmoothed images
      stack :: boolean
        If True, also generate a stacked image (no weighting)
        and compute te, etc.
      smogrid_start :: scalar
        The starting velocity of the re-gridded velocity axis.
        If None, the data aren't regridded, except for the stacked
        spectrum which uses the parameters of the worst resolution
        spectral window.
      smogrid_res :: scalar
        The resolution of the smoothed and re-gridded velocity axis.
        If None, the data aren't regridded, except for the stacked
        spectrum which uses the parameters of the worst resolution
        spectral window.
      smogrid_end :: scalar
        The ending velocity of the re-gridded velocity axis.
        If None, the data aren't regridded, except for the stacked
        spectrum which uses the parameters of the worst resolution
        spectral window.

    Returns: Nothing
    """
    linetype = 'clean'
    if uvtaper: linetype += '.uvtaper'
    linetype += '.pbcor'
    if imsmooth: linetype += '.imsmooth'
    #
    # Generate smoothed velocity axis
    #
    smogrid_velocity = None
    if smogrid_res is not None:
        smogrid_velocity = np.arange(smogrid_start, smogrid_end+smogrid_res,
                                     smogrid_res)
    #
    # Loop over spectral windows
    #
    goodplots = []
    for spw in spws.split(','):
        print("Working on spw {0}".format(spw))
        #
        # Process the data
        #
        mfsimage = '{0}.spw{1}.I.mfs.{2}.image.fits'.format(field, spw, linetype)
        if not os.path.exists(mfsimage):
            print("{0} not found".format(mfsimage))
            continue
        plots = process(field, spw, uvtaper=uvtaper, imsmooth=imsmooth,
                        smogrid_velocity=smogrid_velocity)
        # goodplots += plots
    #
    # Stack the data without weighting
    #
    if stack:
        print("Working on stacked data")
        #
        # Read MFS image, residual image, PB image, and channel image
        #
        mfsdata = []
        chandata = []
        maskdata = []
        freqs = []
        for spw in spws.split(','):
            # get MFS data
            mfsimage = '{0}.spw{1}.I.mfs.{2}.image.fits'.format(field,spw,linetype)
            if not os.path.exists(mfsimage):
                continue
            mfshdu = fits.open(mfsimage)
            mymfsdata = mfshdu[0].data
            mymfsdata[mymfsdata == 0.] = np.nan
            mfsdata.append(mymfsdata)
            # Get channel data
            chanimage = '{0}.spw{1}.I.channel.{2}.image.fits'.format(field,spw,linetype)
            if smogrid_velocity is not None:
                chanimage = '{0}.spw{1}.I.channel.{2}.smogrid.image.fits'.format(field,spw,linetype)
            chanhdu = fits.open(chanimage)
            mychandata = chanhdu[0].data
            mychandata[mychandata == 0.] = np.nan
            chandata.append(chanhdu[0].data)
            freqs.append(mfshdu[0].header['RESTFRQ'])
            # get mask data
            fullrgn = '*.notaper.spw{0}.fullrgn.fits'.format(spw)
            if uvtaper: fullrgn = '*.uvtaper.spw{0}.fullrgn.fits'.format(spw)
            if imsmooth: fullrgn = '*.imsmooth.spw{0}.fullrgn.fits'.format(spw)
            if uvtaper and imsmooth: fullrgn = '*.uvtaper.imsmooth.spw{0}.fullrgn.fits'.format(spw)
            masks = glob.glob(fullrgn)
            cliprgn = np.zeros(mfshdu[0].data.shape, dtype=np.bool)
            for mask in masks:
                fullrgnhdu = fits.open(mask)
                fullrgn_data = np.array(fullrgnhdu[0].data,dtype=np.bool)
                cliprgn = cliprgn | fullrgn_data
            maskdata.append(cliprgn)
        #
        # Create average and save to file
        #
        freqavg = np.mean(freqs)
        # MFS
        mfsavg = np.mean(mfsdata, axis=0)
        mfsavgimage = '{0}.all.I.mfs.{1}.image.fits'.format(field,linetype)
        mfshdu[0].data = mfsavg
        mfshdu[0].header['RESTFRQ'] = freqavg
        mfshdu.writeto(mfsavgimage, overwrite=True, output_verify='fix+warn')
        # Channel
        chanavg = np.mean(chandata, axis=0)
        chanavgimage = '{0}.all.I.channel.{1}.image.fits'.format(field,linetype)
        chanhdu[0].data = chanavg
        chanhdu[0].header['RESTFRQ'] = freqavg
        chanhdu.writeto(chanavgimage, overwrite=True, output_verify='fix+warn')
        # Mask
        allmask = np.sum(maskdata, axis=0, dtype=np.bool).astype(np.int)
        chanavgimage = field
        if uvtaper:
            chanavgimage += '.uvtaper'
        else:
            chanavgimage += '.notaper'
        if imsmooth: chanavgimage += '.imsmooth'
        chanavgimage += '.all.fullrgn.fits'
        mfshdu[0].data = allmask
        mfshdu.writeto(chanavgimage, overwrite=True, output_verify='fix+warn')
        #
        # Process the stacked images
        #
        plots = process(field, 'all', uvtaper=uvtaper, imsmooth=imsmooth,
                        smogrid_velocity=None)
        goodplots += plots
    #
    # Generate PDFs of plots
    #
    all_plots = []
    for plot in goodplots:
        plottype = plot[0]
        spw = plot[1]
        fitsfname = plot[2]
        e_fitsfname = None
        if len(plot) > 3:
            e_fitsfname = plot[3]
        #
        # Read FITS file and generate WCS
        #
        hdu = fits.open(fitsfname)[0]
        wcs = WCS(hdu.header)
        if e_fitsfname is not None:
            e_hdu = fits.open(e_fitsfname)[0]
        #
        # Determine properties of this image
        #
        if plottype == 'linesum':
            title = 'RRL Moment 0 ({0})'.format(spw)
            label = 'Integrated RRL Flux (mJy/beam * km/s)'
            hdu.data *= 1000.
            vlim = (np.nanmin(hdu.data),np.nanmax(hdu.data))
        elif plottype == 'specrms':
            title = 'Spectral RMS ({0})'.format(spw)
            label = 'Flux Density (mJy/beam)'
            hdu.data *= 1000.
            vlim = (np.nanmin(hdu.data),np.nanmax(hdu.data))
        elif plottype == 'lineflux':
            title = 'Line Intensity ({0})'.format(spw)
            label = 'Flux Density (mJy/beam)'
            hdu.data *= 1000.
            vlim = (np.nanmin(hdu.data),np.nanmax(hdu.data))
            e_title = 'Line Intensity Error'
            e_label = 'Flux Density Error (mJy/beam)'
            e_hdu.data *= 1000.
            e_vlim = (np.nanmin(e_hdu.data),np.nanmax(e_hdu.data))
        elif plottype == 'linevlsr':
            title = 'Line LSR Velocity ({0})'.format(spw)
            label = 'LSR Velocity (km/s)'
            vlim = (np.nanmin(hdu.data),np.nanmax(hdu.data))
            e_title = 'Line LSR Velocity Error ({0})'.format(spw)
            e_label = 'LSR Velocity Error (km/s)'
            e_vlim = (np.nanmin(e_hdu.data),np.nanmax(e_hdu.data))
        elif plottype == 'linefwhm':
            title = 'Line FWHM ({0})'.format(spw)
            label = 'FWHM (km/s)'
            vlim = (np.nanmin(hdu.data),np.nanmax(hdu.data))
            e_title = 'Line FWHM Error ({0})'.format(spw)
            e_label = 'FWHM (km/s)'
            e_vlim = (np.nanmin(e_hdu.data),np.nanmax(e_hdu.data))
        elif plottype == 'tesum':
            title = 'Elec. Temp. (Mom 0; {0})'.format(spw)
            label = 'Electron Temperature (K)'
            vlim = (np.nanmin(hdu.data),np.nanmax(hdu.data))
            e_title = 'Elec. Temp. Error (Mom 0; {0})'.format(spw)
            e_label = 'Electron Temperature (K)'
            e_vlim = (np.nanmin(e_hdu.data),np.nanmax(e_hdu.data))            
        elif plottype == 'tefit':
            title = 'Elec. Temp. (Fits; {0})'.format(spw)
            label = 'Electron Temperature (K)'
            vlim = (np.nanmin(hdu.data),np.nanmax(hdu.data))
            e_title = 'Elec. Temp. Error (Fits; {0})'.format(spw)
            e_label = 'Electron Temperature (K)'
            e_vlim = (np.nanmin(e_hdu.data),np.nanmax(e_hdu.data))
        #
        # Generate figure
        #
        plt.ioff()
        fig = plt.figure()
        ax = plt.subplot(projection=wcs.sub(['celestial']))
        ax.set_title(title)
        cax = ax.imshow(hdu.data,origin='lower',interpolation='none',
                        cmap='viridis',vmin=vlim[0],vmax=vlim[1])
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Declination (J2000')
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.set_label(label)
        fname = fitsfname.replace('.fits','.pdf')
        fig.tight_layout()
        fig.savefig(fname,bbox_inches='tight')
        plt.close(fig)
        #
        # Generate uncertainty figure
        #
        if e_fitsfname is not None:
            fig = plt.figure()
            ax = plt.subplot(projection=wcs.sub(['celestial']))
            ax.set_title(e_title)
            cax = ax.imshow(e_hdu.data,origin='lower',interpolation='none',
                            cmap='viridis',vmin=e_vlim[0],vmax=e_vlim[1])
            ax.coords[0].set_major_formatter('hh:mm:ss')
            ax.set_xlabel('RA (J2000)')
            ax.set_ylabel('Declination (J2000')
            cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
            cbar.set_label(e_label)
            e_fname = e_fitsfname.replace('.fits','.pdf')
            fig.tight_layout()
            fig.savefig(e_fname,bbox_inches='tight')
            plt.close(fig)
        #
        # Save plot filenames for master PDF
        #
        all_plots.append(fname)
        if e_fitsfname is not None:
            all_plots.append(e_fname)
    #
    # Generate PDF
    #
    if len(all_plots) > 0:
        fname = field
        if uvtaper: fname += '.uvtaper'
        if imsmooth: fname += '.imsmooth'
        fname += '.teimages.pdf'
        plotfiles = ['{'+fn.replace('.pdf','')+'}.pdf' for fn in all_plots]
        with open(fname,'w') as f:
            f.write(r"\documentclass{article}"+"\n")
            f.write(r"\usepackage{graphicx}"+"\n")
            f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
            f.write(r"\begin{document}"+"\n")
            for i in range(0,len(plotfiles),6):
                f.write(r"\begin{figure}"+"\n")
                f.write(r"\centering"+"\n")
                f.write(r"\includegraphics[width=0.45\textwidth]{"+plotfiles[i]+"}\n")
                if len(plotfiles) > i+1: f.write(r"\includegraphics[width=0.45\textwidth]{"+plotfiles[i+1]+r"} \\"+"\n")
                if len(plotfiles) > i+2: f.write(r"\includegraphics[width=0.45\textwidth]{"+plotfiles[i+2]+"}\n")
                if len(plotfiles) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+plotfiles[i+3]+r"} \\"+"\n")
                if len(plotfiles) > i+4: f.write(r"\includegraphics[width=0.45\textwidth]{"+plotfiles[i+4]+"}\n")
                if len(plotfiles) > i+5: f.write(r"\includegraphics[width=0.45\textwidth]{"+plotfiles[i+5]+"}\n")
                f.write(r"\end{figure}"+"\n")
                f.write(r"\clearpage"+"\n")
            f.write(r"\end{document}")
        os.system('pdflatex -interaction=batchmode {0}'.format(fname))
