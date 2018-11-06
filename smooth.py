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
"""

import __main__ as casa
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import logging.config
import ConfigParser
import shutil

__version__ = "1.0"

# load logging configuration file
logging.config.fileConfig('logging.conf')

def setup(config=None):
    """
    Perform setup tasks: find line and continuum spectral windows

    Inputs: config
      config :: a ConfigParser objet
        The ConfigParser object for this project

    Returns: my_cont_spws, my_line_spws
      my_cont_spws :: string
        comma-separated string of continuum spws
      my_line_spws :: string
        comma-separated string of line spws
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # check config
    #
    if config is None:
        logger.critical("Error: Need to supply a config")
        raise ValueError("Config is None") 
    #
    # Get continuum and line spws from configuration file
    #
    my_cont_spws = config.get("Spectral Windows","Continuum")
    my_line_spws = config.get("Spectral Windows","Line")
    logger.info("Found continuum spws: {0}".format(my_cont_spws))
    logger.info("Found line spws: {0}".format(my_line_spws))
    return (my_cont_spws,my_line_spws)

def smooth_all(field,spws='',config=None,overwrite=False,
               imagetype='clean'):
    """
    Smooth all line and continuum images/cubes to largest 
    beam size of any individual image/cube.

    Inputs: field, spws, config, overwrite, imagetype
      field :: string
        The field to analyze
      spws :: string
        comma separated string of spws to smooth
      config :: a ConfigParser object
        The ConfigParser object for this project
      overwrite :: boolean
        if True, overwrite existing images
      imagetype :: string
        What images to process. For example,
        'dirty', 'clean', 'dirty.uvtaper', or 'clean.uvtaper'

    Returns: Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # check config
    #
    if config is None:
        logger.critical("Error: Need to supply a config")
        raise ValueError("Config is None")
    myspws = ['spw{0}'.format(spw) if spw != 'cont' else spw for spw in spws.split(',')]
    #
    # Find beam major axes, minor axes, and position angles for all
    # available images
    #
    logger.info("Finding largest synthesized beam")
    bmajs = []
    bmins = []
    # images
    images = ['{0}.{1}.mfs.{2}.image.fits'.format(field,spw,imagetype)
              for spw in myspws if os.path.exists('{0}.{1}.mfs.{2}.image.fits'.format(field,spw,imagetype))]
    # primary-beam images
    pbimages = [image.replace('.image.fits','.pbcor.image.fits') for image in images]
    # residual images
    resimages = [image.replace('.image.fits','.residual.fits') for image in images]
    # cubes
    cubes = ['{0}.{1}.channel.{2}.image.fits'.format(field,spw,imagetype)
             for spw in myspws if os.path.exists('{0}.{1}.channel.{2}.image.fits'.format(field,spw,imagetype))]
    # primary beam cubes
    pbcubes = [cube.replace('.image.fits','.pbcor.image.fits') for cube in cubes]
    # residual cubes
    rescubes = [cube.replace('.image.fits','.residual.fits') for cube in cubes]
    for imagename in images+cubes:
        if 'perplanebeams' in casa.imhead(imagename).keys():
            beams = casa.imhead(imagename)['perplanebeams']['beams']
            bmajs.append(np.max([beams[key]['*0']['major']['value'] for key in beams.keys()]))
            bmins.append(np.max([beams[key]['*0']['major']['value'] for key in beams.keys()]))
        else:
            bmajs.append(casa.imhead(imagename,mode='get',
                         hdkey='beammajor')['value'])
            bmins.append(casa.imhead(imagename,mode='get',
                         hdkey='beamminor')['value'])
    #
    # Smooth available images to maximum (circular) beam size
    # + 0.1 pixel size (otherwise imsmooth will complain)
    #
    cell_size = float(config.get("Clean","cell").replace('arcsec',''))
    bmaj_target = np.max(bmajs)+0.1*cell_size
    bmin_target = np.max(bmajs)+0.1*cell_size
    bpa_target = 0.
    logger.info("Smoothing all images to")
    logger.info("Major axis: {0} arcsec".format(bmaj_target))
    logger.info("Minor axis: {0} arcsec".format(bmin_target))
    logger.info("Position angle: {0} degs".format(bpa_target))
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
    logger.info("Done!")

def main(field,config_file='',overwrite=False,imagetype='clean'):
    """
    Smooth all images, including continuum MFS, to common beam

    Inputs: field, spws, config, overwrite, imagetype
      field :: string
        The field to analyze
      config_file :: string
        The filename of the configuration field for this project
      overwrite :: boolean
        if True, overwrite existing images
      imagetype :: string
        What images to process. For example,
        'dirty', 'clean', 'dirty.uvtaper', or 'clean.uvtaper'

    Returns: Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Check inputs
    #
    if not os.path.exists(config_file):
        logger.critical('Configuration file not found')
        raise ValueError('Configuration file not found!')
    #
    # load configuration file
    #
    config = ConfigParser.ConfigParser()
    logger.info("Reading configuration file {0}".format(config_file))
    config.read(config_file)
    logger.info("Done.")
    #
    # initial setup
    #
    my_cont_spws,my_line_spws = setup(config=config)
    #
    # Smooth all MFS and channel images to common beam
    #
    all_spws = ','.join(['cont',my_cont_spws,my_line_spws])
    smooth_all(field,spws=all_spws,config=config,
               overwrite=overwrite,imagetype=imagetype)
