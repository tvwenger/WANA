"""
gen_regions_linmos.py - WANA region generation script

Generate a polygon region containing all emission associated with
a given peak region.

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
   Update to WISP V2.0 including Stokes and mosaics
   Compute watershed regions based on image_data * distance from
   background to handle very confused regions.
"""

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy import ndimage as nd
from scipy.signal import convolve2d
from skimage.morphology import watershed
from skimage.draw import ellipse

__version__ = "1.0"

def main(field,spws,stokes='I',taper=False,imsmooth=False,
         sigmaclip=3.):
    """
    Given some peak region, generate a polygon region containing
    all emission associated with that peak. We do so by starting
    at the peak and working down in flux density until we either
    reach the noise floor or a nearby region.
    The best algorithm to use is watershed segmentation.
    These region files are saved as fits images with names like
    <region file without .rgn>.fullrgn.fits

    Inputs:
      field :: string
        the field name.
      spws :: string
        comma-separated string of spws for which to generate regions
      stokes :: string
        The stokes parameters in the image
      regions :: list of strings
        peak region files
      taper :: boolean
        if True, use uvtaper images
      imsmooth :: boolean
        if True, use imsmooth images
      sigmaclip :: scalar
        at what sigma level to set image noise floor

    Returns: Nothing
    """
    #
    # Read coordinates of peak regions
    #
    rgnend = '.notaper'
    if taper: rgnend = '.uvtaper'
    if imsmooth: rgnend += '.imsmooth'
    rgnend += '.rgn'
    peakregions = glob.glob('*{0}'.format(rgnend))
    peakregions.sort()
    peak_positions = []
    for peakregion in peakregions:
        # read second line in region file
        with open(peakregion,'r') as f:
            f.readline()
            data = f.readline()
            splt = data.split(' ')
            RA = splt[1].replace('[[','').replace(',','')
            RA_h, RA_m, RA_s = RA.split(':')
            RA = '{0}h{1}m{2}s'.format(RA_h,RA_m,RA_s)
            dec = splt[2].replace('],','')
            dec_d, dec_m, dec_s, dec_ss = dec.split('.')
            dec = '{0}d{1}m{2}.{3}s'.format(dec_d,dec_m,dec_s,dec_ss)
            coord = SkyCoord(RA,dec,frame='fk5')
            peak_positions.append(coord)
    #
    # Loop over each spw
    #
    for spw in spws.split(','):
        #
        # Open images, generate WCS
        #
        if spw != 'cont':
            spw = 'spw{0}'.format(spw)
        #
        # Read image. This is primary beam corrected.
        #
        image = '{0}.{1}.{2}.mfs.clean'.format(field, spw, stokes)
        if taper: image += '.uvtaper'
        if imsmooth: image += '.imsmooth'
        image += '.image.linmos.fits'
        if not os.path.exists(image):
            continue
        image_hdu = fits.open(image)[0]
        image_wcs = WCS(image_hdu.header)
        wcs_celest = image_wcs.sub(['celestial'])
        #
        # Read residual
        #
        residual = '{0}.{1}.{2}.mfs.clean'.format(field, spw, stokes)
        if taper: residual += '.uvtaper'
        if imsmooth: residual += '.imsmooth'
        residual += '.residual.linmos.fits'
        residual_hdu = fits.open(residual)[0]
        #
        # Read mosaic weights (primary beam) image
        #
        pb = '{0}.{1}.{2}.mfs.pb.linmos.fits'.format(field, spw, stokes)
        pb_hdu = fits.open(pb)[0]
        #
        # Get un-primary beam corrected image and residuals
        #
        image_data = image_hdu.data[0,0]*pb_hdu.data[0,0]
        residual_data = residual_hdu.data[0,0]*pb_hdu.data[0,0]
        #
        # Clip image at sigmaclip*rms. The noise can vary across the
        # mosaic, so this is only an approximation
        #
        rms = 1.4826*np.nanmedian(np.abs(residual_data-np.nanmedian(residual_data)))
        clip = image_data < sigmaclip*rms
        image_clip = np.ones(image_data.shape, dtype=int)
        image_clip[clip] = 0
        #
        # Compute pixel positions of peak regions
        # N.B. Flip to match WCS
        #
        peak_pixels = []
        for peak_position in peak_positions:
            pixel = wcs_celest.wcs_world2pix(peak_position.ra.deg,peak_position.dec.deg,1)
            peak_pixels.append((int(pixel[1]),int(pixel[0])))
        #
        # Create seed array for watershed
        #
        seed_locations = np.zeros(image_data.shape,dtype=int)
        for i,peak_pixel in enumerate(peak_pixels):
            seed_locations[peak_pixel[0],peak_pixel[1]] = i+1
        #
        # Compute distance to background
        #
        distance = nd.distance_transform_edt(image_clip)
        #
        # Perform watershed segmentation
        #
        result = watershed(-distance*image_data, seed_locations,
                           mask=image_clip)
        #
        # For each peak region, find associated watershed region
        # and save as boolean mask fits image
        #
        for i,peakregion in enumerate(peakregions):
            #
            # Generate full region mask fits image
            #
            fullregion = peakregion.replace('.rgn','.{0}.fullrgn.fits'.format(spw))
            image_hdu.data[0,0] = result == i+1
            image_hdu.writeto(fullregion,overwrite=True)
