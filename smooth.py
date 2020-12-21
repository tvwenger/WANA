"""
smooth.py - WANA image smoothing program

Smooth several images or data cubes to a common beam size.

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
    Update for WISP V2.0 support (stokes image names, mosaic)
"""

import __main__ as casa
import os
import numpy as np
from astropy.io import fits

__version__ = "2.0"

def smooth_all(field, spws='', stokes='', imagetype='clean',
               mosaic=False, overwrite=False):
    """
    Smooth all line and continuum images/cubes to largest 
    beam size of any individual image/cube.

    Inputs: 
      field :: string
        The field to analyze
      spws :: string
        comma separated string of spws to smooth
      stokes :: string
        The Stokes parameters in the image
      imagetype :: string
        What images to process. For example,
        'dirty', 'clean', 'dirty.uvtaper', or 'clean.uvtaper'
      mosaic :: boolean
        if True, these are mosaic images (.linmos.fits)
      overwrite :: boolean
        if True, overwrite existing images

    Returns: Nothing
    """
    myspws = ['spw{0}'.format(spw) if spw != 'cont' else spw
              for spw in spws.split(',')]
    #
    # Find beam major axes, minor axes, and position angles for all
    # available images
    #
    print("Finding largest synthesized beam")
    bmajs = []
    bmins = []
    # images
    imname = '{0}.{1}.{2}.mfs.{3}.image.fits'
    if mosaic:
        imname = '{0}.{1}.{2}.mfs.{3}.image.linmos.fits'
    images = [imname.format(field, spw, stokes, imagetype)
              for spw in myspws
              if os.path.exists(imname.format(field,spw,stokes,imagetype))]
    # residual images
    resimages = [image.replace('.image.', '.residual.') for image in images]
    # cubes
    imname = '{0}.{1}.{2}.channel.{3}.image.fits'
    if mosaic:
        imname = '{0}.{1}.{2}.channel.{3}.image.linmos.fits'
    cubes = [imname.format(field, spw, stokes, imagetype)
             for spw in myspws
             if os.path.exists(imname.format(field,spw,stokes,imagetype))]
    # residual cubes
    rescubes = [image.replace('.image.', '.residual.') for image in cubes]
    for imagename, resimagename in zip(images+cubes, resimages+rescubes):
        with fits.open(imagename) as imhdulist:
            bunit = imhdulist[0].header['BUNIT']
            if len(imhdulist) > 1:
                bmaj = imhdulist[1].data['BMAJ'][0] # arcsec
                bmin = imhdulist[1].data['BMIN'][0] # arcsec
                bpa = imhdulist[1].data['BPA'][0]
            else:
                bmaj = imhdulist[0].header['BMAJ'] * 3600. # arcsec
                bmin = imhdulist[0].header['BMIN'] * 3600. # arcsec
                bpa = imhdulist[0].header['BPA']
            # check that residual images have beams and units
            with fits.open(resimagename, 'update') as reshdulist:
                if 'BMIN' not in reshdulist[0].header:
                    reshdulist[0].header['BMIN'] = bmin/3600. # deg
                    reshdulist[0].header['BMAJ'] = bmaj/3600. # deg
                    reshdulist[0].header['BPA'] = bpa
                if not reshdulist[0].header['BUNIT']:
                    reshdulist[0].header['BUNIT'] = bunit
            # check that residual images have units
            # append
            bmajs.append(bmaj)
            bmins.append(bmin)
    #
    # Smooth available images to maximum (circular) beam size
    # + pixel diagonal size (otherwise imsmooth will complain)
    #
    cell_size = abs(casa.imhead(imagename)['incr'][0]) * 206265.
    bmaj_target = np.max(bmajs)+1.42*cell_size
    bmin_target = np.max(bmajs)+1.42*cell_size
    bpa_target = 0.
    print("Smoothing all images to")
    print("Major axis: {0:.2f} arcsec".format(bmaj_target))
    print("Minor axis: {0:.2f} arcsec".format(bmin_target))
    print("Position angle: {0:.2f} degs".format(bpa_target))
    bmaj_target = {'unit':'arcsec','value':bmaj_target}
    bmin_target = {'unit':'arcsec','value':bmin_target}
    bpa_target = {'unit':'deg','value':bpa_target}
    for imagename, resimagename in zip(images+cubes, resimages+rescubes):
        # export velocity axis if this is a cube
        velocity = 'channel' in imagename
        # smooth image
        outfile = imagename.replace('.image.fits','.imsmooth.image')
        if mosaic:
            outfile = imagename.replace('.image.linmos.fits', '.imsmooth.image.linmos')
        casa.imsmooth(imagename=imagename,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        velocity=velocity,overwrite=True,history=False)
        # primary beam correct
        if not mosaic:
            smoimagename = outfile
            pbimage = imagename.replace('.{0}.image.fits'.format(imagetype), '.pb.fits')
            outfile = imagename.replace('.image.fits', '.pbcor.imsmooth.image')
            casa.impbcor(imagename=smoimagename, pbimage=pbimage,
                        outfile=outfile, overwrite=True)
            casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                            velocity=velocity,overwrite=True,history=False)
        # check that residual image has beam size, if not add it
        with fits.open(resimagename, 'update') as hdulist:
            hdu = hdulist[0]
            if 'BMIN' not in hdu.header:
                hdu.header['BMIN'] = bmin_target['value']/3600.
                hdu.header['BMAJ'] = bmaj_target['value']/3600.
                hdu.header['BPA'] = bpa_target['value']
        # smooth residual image
        outfile = resimagename.replace('.residual.fits','.imsmooth.residual')
        if mosaic:
            outfile = resimagename.replace('.residual.linmos.fits', '.imsmooth.residual.linmos')
        casa.imsmooth(imagename=resimagename,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        velocity=velocity,overwrite=True,history=False)
    print("Done!")

def main(field, spws, stokes='I', imagetype='clean', mosaic=False,
         overwrite=False,):
    """
    Smooth all MFS and cube spws to a common beam size.

    Inputs:
      field :: string
        The field to analyze
      spws :: string
        Comma-separated list of spectral window images to smooth
      stokes :: string
        The stokes parameters in the images
      imagetype :: string
        What images to process. For example,
        'dirty', 'clean', 'dirty.uvtaper', or 'clean.uvtaper'
      mosaic :: boolean
        if True, these are mosaic images (.linmos.fits)
      overwrite :: boolean
        if True, overwrite existing images

    Returns: Nothing
    """
    #
    # Smooth all MFS and channel images to common beam
    #
    smooth_all(field, spws=spws, stokes=stokes, imagetype=imagetype,
               mosaic=mosaic, overwrite=overwrite)
