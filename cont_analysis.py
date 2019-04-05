"""
cont_analysis.py - WANA continuum image analysis program

Analyze continuum images, compute flux densities, etc.

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
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

__version__ = "1.0"

def main(field,regions,spws,
         fluxtype='peak',taper=False,imsmooth=False,
         outfile='cont_info.txt'):
    """
    Measure continuum flux density and rms for each source in the
    continuum MFS images. Generate an output table with the measured
    parameters.

    Inputs:
      field :: string
        The field name
      region :: list of strings
        If only one element, the region file used to analyze each
        spw. Otherwise, the region file to use for each spw.
      spws :: string
        comma-separated string of spws to analyze
      fluxtype :: string
        What type of flux to measure. 'peak' to use peak regions and
        measure peak flux density, 'total' to use full regions and
        measure total flux density.
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
        # 0    1           2          3        4           5          6           7
        # spw  frequency   cont       rms      area_arcsec area_pixel beam_arcsec beam_pixel
        # #    MHz         mJy/beam   mJy/beam arcsec2     pixels     arcsec2     pixels
        # cont 9494.152594 12345.6789 123.4567 12345.67    123456.78  12345.67    123456.78
        # 1    9494.152594 12345.6789 123.4567 12345.67    123456.78  12345.67    123456.78
        # 1234 12345678902 1234567890 12345678 12345678901 1234567890 12345678901 1234567890
        #
        headerfmt = '{0:4} {1:12} {2:10} {3:8} {4:11} {5:10} {6:11} {7:10}\n'
        rowfmt = '{0:4} {1:12.6f} {2:10.4f} {3:8.4f} {4:11.2f} {5:10.2f} {6:11.2f} {7:10.2f}\n'
        f.write(headerfmt.format('spw','frequency','cont','rms','area_arcsec','area_pixel','beam_arcsec','beam_pixel'))
        if fluxtype == 'total':
            fluxunit = 'mJy'
        else:
            fluxunit = 'mJy/beam'
        f.write(headerfmt.format('#','MHz',fluxunit,fluxunit,'arcsec2','pixels','arcsec2','pixels'))
        #
        # Read data for each spw
        #
        for spw,region in zip(spws,regions):
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
            # Read region file, extract data from region center pixel
            #
            if not os.path.exists(region):
                cont = np.nan
                rms = np.nan
                area_pixel = np.nan
                area_arcsec = np.nan
            elif fluxtype == 'peak':
                with open(region,'r') as freg:
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
                area_pixel = 1.
                area_arcsec = pixel_size
                pix = wcs_celest.wcs_world2pix(coord.ra.deg,coord.dec.deg,1)
                cont = 1000.*image_hdu.data[0,0,int(pix[1]),int(pix[0])] # mJy/beam
                #
                # Compute primary beam corrected rms at peak location
                #
                rms = 1000.*image_rms[int(pix[1]),int(pix[0])]
            #
            # Read region file, sum spectrum region
            #
            else:
                region_mask = np.array(fits.open(region)[0].data[0,0],dtype=np.bool)
                area_pixel = np.sum(region_mask)
                area_arcsec = area_pixel * pixel_size
                #
                # Extract flux, mJy/beam*pixels/beam_area_pixels
                #
                cont = 1000.*np.sum(image_hdu.data[0,0,region_mask])/beam_pixel # mJy
                #
                # Compute primary beam corrected rms
                #
                rms = 1000.*np.sqrt(np.sum(image_rms[region_mask]**2.))/beam_pixel # mJy
                # spw  frequency   cont       rms      area_arcsec area_pixel beam_arcsec beam_pixel
            f.write(rowfmt.format(spw.replace('spw',''),frequency,cont,rms,area_arcsec,area_pixel,beam_arcsec,beam_pixel))
