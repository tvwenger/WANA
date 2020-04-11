"""
stokes_analysis.py - WANA continuum stokes image analysis program

Analyze continuum stokes images

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

Trey V. Wenger September 2019 - V2.0
    Update to WISP V2.0 to handle Stokes images
"""

import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

__version__ = "1.0"

def main(field,regions,spws,label,
         stokes='IQUV',taper=False,imsmooth=False,
         outfile='stokes_info.txt'):
    """
    Measure stokes brightnesses and rms

    Inputs:
      field :: string
        The field name
      regions :: list of strings
        If only one element, the region file used to analyze each
        spw. Otherwise, the region file to use for each spw.
      spws :: string
        comma-separated string of spws to analyze
      label :: string
        A label to add to the filename
      stokes :: string
        The stokes parameters in the image
      taper :: boolean
        if True, use uv-tapered images
      imsmooth :: boolean
        if True, use imsmooth images
      outfile :: string
        Filename where the output table is written.

    Returns: Nothing
    """
    #
    # If supplied only one region, expand to all spws
    #
    spws = ['spw{0}'.format(spw) if spw != 'cont' else 'cont'
            for spw in spws.split(',')]
    if len(regions) == 1:
        regions = [regions[0] for spw in spws]
    #
    # Set-up file
    #
    with open(outfile,'w') as f:
        # 0    1           2      3          4        5        6        7        8        9        10       11          12         13          14         15
        # spw  frequency   stokes cont       e_cont   map_mean map_med  map_rms  reg_mean reg_med  reg_rms  area_arcsec area_pixel beam_arcsec beam_pixel pb_level
        # #    MHz                mJy        mJy      mJy/beam mJy/beam mJy/beam mJy/beam mJy/beam mJy/beam arcsec2     pixels     arcsec2     pixels     %
        # cont 9494.152594 I      12345.6789 123.4567 1234.567 1234.567 1234.567 1234.567 1234.567 1234.567 12345.67    123456.78  12345.67    123456.78  100.00
        # 1    9494.152594 Q      12345.6789 123.4567 1234.567 1234.567 1234.567 1234.567 1234.567 1234.567 12345.67    123456.78  12345.67    123456.78  100.00
        # 1234 12345678902 123456 1234567890 12345678 12345678 12345678 12345678 12345678 12345678 12345678 12345678901 1234567890 12345678901 1234567890 12345678
        #
        headerfmt = '{0:4} {1:12} {2:6} {3:10} {4:8} {5:8} {6:8} {7:8} {8:8} {9:8} {10:8} {11:11} {12:10} {13:11} {14:10} {15:8}\n'
        rowfmt = '{0:4} {1:12.6f} {2:6} {3:10.4f} {4:8.4f} {5:8.3f} {6:8.3f} {7:8.3f} {8:8.3f} {9:8.3f} {10:8.3f} {11:11.2f} {12:10.2f} {13:11.2f} {14:10.2f} {15:8.2f}\n'
        f.write(headerfmt.format('spw','frequency','stokes','cont','e_cont','map_mean','map_med','map_rms','reg_mean','reg_med','reg_rms','area_arcsec','area_pixel','beam_arcsec','beam_pixel','pb_level'))
        f.write(headerfmt.format('#','MHz','','mJy','mJy','mJy/beam','mJy/beam','mJy/beam','mJy/beam','mJy/beam','mJy/beam','arcsec2','pixels','arcsec2','pixels','%'))
        #
        # Read data for each spw
        #
        for spw,region in zip(spws,regions):
            #
            # Read image
            #
            image = '{0}.{1}.{2}.mfs.clean'.format(field, spw, stokes)
            if taper: image += '.uvtaper'
            if imsmooth: image += '.imsmooth'
            image += '.image.fits'
            if not os.path.exists(image):
                print("{0} not found.".format(image))
                continue
            image_hdulist = fits.open(image)
            image_hdu = image_hdulist[0]
            #
            # Read pbcor image
            #
            image = '{0}.{1}.{2}.mfs.clean'.format(field, spw, stokes)
            if taper: image += '.uvtaper'
            image += '.pbcor'
            if imsmooth: image += '.imsmooth'
            image += '.image.fits'
            if not os.path.exists(image):
                print("{0} not found.".format(image))
                continue
            pbcor_hdu = fits.open(image)[0]
            #
            # Read residual image
            #
            image = '{0}.{1}.{2}.mfs.clean'.format(field, spw, stokes)
            if taper: image += '.uvtaper'
            if imsmooth: image += '.imsmooth'
            image += '.residual.fits'
            residual_hdu = fits.open(image)[0]
            #
            # Get WCS
            #
            image_wcs = WCS(image_hdu.header)
            wcs_celest = image_wcs.sub(['celestial'])
            frequency = image_hdu.header['CRVAL3']/1.e6 # MHz
            #
            # Calculate beam area and pixel size
            #
            pixel_size = 3600.**2. * np.abs(image_hdu.header['CDELT1'] * image_hdu.header['CDELT2']) # arcsec^2
            if 'BMAJ' in image_hdu.header.keys():
                beam_maj = image_hdu.header['BMAJ'] # deg
                beam_min = image_hdu.header['BMIN'] # deg
                beam_pa = image_hdu.header['BPA'] # deg
            elif len(image_hdulist) > 1:
                hdu = image_hdulist[1]
                convert = 1.
                # convert arcsec to deg if necessary
                if 'arcsec' in hdu.header['TUNIT1']:
                    convert = 1./3600.
                beam_maj = convert*hdu.data['BMAJ'][0] # deg
                beam_min = convert*hdu.data['BMIN'][0] # deg
                beam_pa = hdu.data['BPA'][0]
            else:
                raise ValueError("Could not get beam size!")
            beam_arcsec = 3600.**2. * np.pi*beam_min*beam_maj/(4.*np.log(2.)) # arcsec^2
            beam_pixel = beam_arcsec / pixel_size
            #
            # Construct synthesized beam kernel 2D Gaussian centered on origin
            # N.B. Fits stored as transpose
            #
            y_size, x_size = image_hdu.data[0,0].shape
            x_axis = np.arange(x_size) - x_size/2.
            y_axis = np.arange(y_size) - y_size/2.
            y_grid, x_grid = np.meshgrid(y_axis, x_axis, indexing='ij')
            lon_grid = -image_hdu.header['CDELT1']*x_grid
            lat_grid = image_hdu.header['CDELT2']*y_grid
            bmin = beam_min/(2.*np.sqrt(2.*np.log(2.)))
            bmaj = beam_maj/(2.*np.sqrt(2.*np.log(2.)))
            bpa = -np.deg2rad(beam_pa)+np.pi/2.
            # 2-D Gaussian parameters
            A = np.cos(bpa)**2./(2.*bmaj**2.) + np.sin(bpa)**2./(2.*bmin**2.)
            B = -np.sin(2.*bpa)/(4.*bmaj**2.) + np.sin(2.*bpa)/(4.*bmin**2.)
            C = np.sin(bpa)**2./(2.*bmaj**2.) + np.cos(bpa)**2./(2.*bmin**2.)
            beam_kernel = np.exp(-(A*lon_grid**2. + 2.*B*lon_grid*lat_grid + C*lat_grid**2.))
            #
            # Get residual image RMS for each stokes
            #
            image_rms = 1.4825*np.nanmedian(np.abs(residual_hdu.data[:,0].T-np.nanmedian(residual_hdu.data[:,0], axis=(1,2))).T, axis=(1,2))
            #
            # Get PB-correctd RMS for each stokes
            #
            pbimage = '{0}.{1}.{2}.mfs.pb.fits'.format(field, spw, stokes)
            pb_hdu = fits.open(pbimage)[0]
            image_rms = (image_rms/pb_hdu.data[:,0].T).T
            image_rms[np.isinf(image_rms)] = np.nan
            #
            # Read region file, extract data from region center pixel
            #
            if not os.path.exists(region):
                area_pixel = np.nan
                area_arsec = np.nan
                cont = [np.nan for ind in stokes]
                e_cont = [np.nan for ind in stokes]
                map_mean = [np.nan for ind in stokes]
                map_med = [np.nan for ind in stokes]
                map_rms = [np.nan for ind in stokes]
                reg_mean = [np.nan for ind in stokes]
                reg_med = [np.nan for ind in stokes]
                reg_rms = [np.nan for ind in stokes]
                pb_level = [np.nan for ind in stokes]
            else:
                region_mask = np.array(fits.open(region)[0].data[0,0],dtype=np.bool)
                area_pixel = np.sum(region_mask)
                area_arcsec = area_pixel * pixel_size
                #
                # Loop over stokes parameters and compute things
                #
                cont = np.zeros(len(stokes))*np.nan
                e_cont = np.zeros(len(stokes))*np.nan
                map_mean = np.zeros(len(stokes))*np.nan
                map_med = np.zeros(len(stokes))*np.nan
                map_rms = np.zeros(len(stokes))*np.nan
                reg_mean = np.zeros(len(stokes))*np.nan
                reg_med = np.zeros(len(stokes))*np.nan
                reg_rms = np.zeros(len(stokes))*np.nan
                pb_level = np.zeros(len(stokes))*np.nan
                for stokesi in range(len(stokes)):
                    #
                    # Compute flux in each stokes, mJy/beam*pixels/beam_area_pixels
                    #
                    cont[stokesi] = 1000.*np.sum(image_hdu.data[stokesi,0,region_mask])/beam_pixel # mJy
                    #
                    # Compute primary beam corrected error formally
                    # taking into consideration correlated noise by
                    # using synthesized beam kernel centered on each pixel
                    # N.B. transpose when rolling beam_kernel
                    #
                    p_sigma = np.array([np.nansum(image_rms[stokesi] * np.roll(beam_kernel, (int(y), int(x)), axis=(0,1)))
                                        for x, y in zip(x_grid[region_mask], y_grid[region_mask])])
                    e_cont[stokesi] = 1000.*np.sqrt(np.sum(image_rms[stokesi,region_mask] * p_sigma))/beam_pixel # mJy
                    #
                    # Compute mean, median, and rms of entire image
                    #
                    map_mean[stokesi] =1000.*np.nanmean(image_hdu.data[stokesi,0])
                    map_med[stokesi] = 1000.*np.nanmedian(image_hdu.data[stokesi,0])
                    map_rms[stokesi] = 1000.*1.4825*np.nanmedian(np.abs(image_hdu.data[stokesi,0]-np.nanmedian(image_hdu.data[stokesi,0])))
                    #
                    # Compute mean, median, and rms within region
                    #
                    reg_mean[stokesi] =1000.*np.nanmean(image_hdu.data[stokesi,0,region_mask])
                    reg_med[stokesi] = 1000.*np.nanmedian(image_hdu.data[stokesi,0,region_mask])
                    reg_rms[stokesi] = 1000.*1.4825*np.nanmedian(np.abs(image_hdu.data[stokesi,0,region_mask]-np.nanmedian(image_hdu.data[stokesi,0,region_mask])))
                    #
                    # Get average primary beam level over region
                    #
                    pb_level[stokesi] = 100.*np.nanmean(pb_hdu.data[stokesi,0,region_mask])
            #
            # Write rows to file
            #
            for stokesi, stoke in enumerate(stokes):
                f.write(rowfmt.format(spw.replace('spw',''),frequency,stoke,
                                      cont[stokesi],e_cont[stokesi],map_mean[stokesi],map_med[stokesi],map_rms[stokesi],
                                      reg_mean[stokesi],reg_med[stokesi],reg_rms[stokesi],
                                      area_arcsec,area_pixel,beam_arcsec,beam_pixel,pb_level[stokesi]))
