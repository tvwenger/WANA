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
    Update for WISP V2.0 support (stokes image names)
"""

import __main__ as casa
import os
import numpy as np

__version__ = "2.0"

def smooth_all(field, spws='', stokes='', imagetype='clean',
               overwrite=False):
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
    images = ['{0}.{1}.{2}.mfs.{3}.image.fits'.format(field, spw, stokes, imagetype)
              for spw in myspws
              if os.path.exists('{0}.{1}.{2}.mfs.{3}.image.fits'.format(field,spw,stokes,imagetype))]
    # primary-beam images
    pbimages = [image.replace('.image.fits','.pbcor.image.fits') for image in images]
    # residual images
    resimages = [image.replace('.image.fits','.residual.fits') for image in images]
    # cubes
    cubes = ['{0}.{1}.{2}.channel.{3}.image.fits'.format(field,spw,stokes,imagetype)
             for spw in myspws if os.path.exists('{0}.{1}.{2}.channel.{3}.image.fits'.format(field,spw,stokes,imagetype))]
    # primary beam cubes
    pbcubes = [cube.replace('.image.fits','.pbcor.image.fits') for cube in cubes]
    # residual cubes
    rescubes = [cube.replace('.image.fits','.residual.fits') for cube in cubes]
    for imagename in images+cubes:
        if 'perplanebeams' in casa.imhead(imagename).keys():
            beams = casa.imhead(imagename)['perplanebeams']['beams']
            bmajs.append(np.max([beams[key]['*0']['major']['value'] for key in beams.keys()]))
            bmins.append(np.max([beams[key]['*0']['minor']['value'] for key in beams.keys()]))
        else:
            bmajs.append(casa.imhead(imagename,mode='get',
                         hdkey='beammajor')['value'])
            bmins.append(casa.imhead(imagename,mode='get',
                         hdkey='beamminor')['value'])
    #
    # Smooth available images to maximum (circular) beam size
    # + 0.1 pixel size (otherwise imsmooth will complain)
    #
    cell_size = abs(casa.imhead(imagename)['incr'][0]) * 206265.
    bmaj_target = np.max(bmajs)+0.1*cell_size
    bmin_target = np.max(bmajs)+0.1*cell_size
    bpa_target = 0.
    print("Smoothing all images to")
    print("Major axis: {0} arcsec".format(bmaj_target))
    print("Minor axis: {0} arcsec".format(bmin_target))
    print("Position angle: {0} degs".format(bpa_target))
    bmaj_target = {'unit':'arcsec','value':bmaj_target}
    bmin_target = {'unit':'arcsec','value':bmin_target}
    bpa_target = {'unit':'deg','value':bpa_target}
    for imagename,pbimagename,resimagename in \
      zip(images+cubes,pbimages+pbcubes,resimages+rescubes):
        # export velocity axis if this is a cube
        velocity = 'channel' in imagename
        # smooth image
        outfile = imagename.replace('.image.fits','.imsmooth.image')
        casa.imsmooth(imagename=imagename,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        velocity=velocity,overwrite=True,history=False)
        # smooth pb image
        outfile = pbimagename.replace('.image.fits','.imsmooth.image')
        casa.imsmooth(imagename=pbimagename,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        velocity=velocity,overwrite=True,history=False)
        # smooth residual image
        outfile = resimagename.replace('.residual.fits','.imsmooth.residual')
        casa.imsmooth(imagename=resimagename,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        velocity=velocity,overwrite=True,history=False)
    print("Done!")

def main(field, spws, stokes='I', imagetype='clean', overwrite=False,):
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
      overwrite :: boolean
        if True, overwrite existing images

    Returns: Nothing
    """
    #
    # Smooth all MFS and channel images to common beam
    #
    smooth_all(field, spws=spws, stokes=stokes, imagetype=imagetype,
               overwrite=overwrite)
