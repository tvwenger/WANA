"""
plot_regions.py - WANA image plotting script (with regions)

Generate figures of images including regions defined either as
- CASA region files
- Boolean masks
- WISE Catalog of Galactic HII Region positions and sizes

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
    Generate CASA region file identifying peaks associated with
    WISE sources.
"""

import os

import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch

__version__ = "1.0"

class HandlerEllipse(HandlerPatch):
    """
    This adds the ability to create ellipses within a matplotlib
    legend. See:
     https://stackoverflow.com/questions/44098362/using-mpatches-patch-for-a-custom-legend
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=height + xdescent,
                    height=height + ydescent, fill=False)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def main(field,spws,stokes='I',
         taper=False,imsmooth=False,
         wisefile=None,levels=[5.,10.,20.,50.,100.]):
    """
    Plot WISE Catalog regions on top of images.
    Background image is primary beam corrected MFS image of each spw.
    Contours are at levels based on noise in non-PB corrected MFS 
    image (i.e. they are true SNR).

    Will also plot .rgn files if they match the following naming
    convention:
    <field>.*.rgn
    <field>.*.fullrgn.fits

    Images are saved as <original without .fits>.reg.pdf
    And compiled in <field>.*.regionplots.pdf
    
    Inputs:
      field :: string
        The field name
      spws :: string
        Comma-separated string of spws to plot.
      stokes :: string
        The Stokes parameters in the image.
      taper :: boolean
        if True, use uvtaper images
      imsmooth :: boolean
        if True, use imsmooth images
      wisefile :: string
        if None, don't plot WISE Catalog positions. Otherwise,
        path to WISE Catalog positions data file with format:
          GName            RA         Dec         Size    
          #                deg        deg         arcsec  
          G000.003+00.127  266.282628  -28.866597   206.70
          ...
          N.B. Size is IR radius
      levels :: list of scalars
        The contour levels as multiplicative factor of RMS noise
        in non-PB corrected MFS image.

    Returns: Nothing
    """
    #
    # Get peak regions
    #
    rgnend = '.notaper'
    if taper:
        rgnend = '.uvtaper'
    if imsmooth:
        rgnend += '.imsmooth'
    rgnend += '.rgn'
    peak_regions = glob.glob('*{0}'.format(rgnend))
    peak_regions.sort()
    #
    # convert levels to array
    #
    levels = np.array(levels)
    #
    # Loop over spws
    #
    outimages = []
    for spw in spws.split(','):
        #
        # Open fits files, generate WCS
        #
        if spw != 'cont':
            spw = 'spw{0}'.format(spw)
        #
        # Read image
        #
        image = '{0}.{1}.{2}.mfs.clean'.format(field, spw, stokes)
        if taper: image += '.uvtaper'
        if imsmooth: image += '.imsmooth'
        image += '.image.fits'
        if not os.path.exists(image):
            continue
        hdulist = fits.open(image)
        image_hdu = hdulist[0]
        #
        # Read image header
        #
        image_wcs = WCS(image_hdu.header)
        wcs_celest = image_wcs.sub(['celestial'])
        #
        # Read pbcor image
        #
        pbcorr = '{0}.{1}.{2}.mfs.clean'.format(field, spw, stokes)
        if taper: pbcorr += '.uvtaper'
        pbcorr += '.pbcor'
        if imsmooth: pbcorr += '.imsmooth'
        pbcorr += '.image.fits'
        pbcorr_hdu = fits.open(pbcorr)[0]
        #
        # Read residual image
        #
        residual = '{0}.{1}.{2}.mfs.clean'.format(field, spw, stokes)
        if taper: residual += '.uvtaper'
        if imsmooth: residual += '.imsmooth'
        residual += '.residual.fits'
        residual_hdu = fits.open(residual)[0]
        #
        # Compute RMS from residuals (MAD)
        #
        sigma = 1.4826*np.nanmedian(np.abs(residual_hdu.data[0,0]-np.nanmedian(residual_hdu.data[0,0])))
        #
        # Generate figure
        #
        plt.ioff()
        fig = plt.figure()
        ax = plt.subplot(projection=wcs_celest)
        ax.set_title(image)
        # image is PB corrected
        cax = ax.imshow(pbcorr_hdu.data[0,0],origin='lower',
                        interpolation='none',cmap='binary')
        # contours are non-PB corrected
        if sigma > 0.:
            con = ax.contour(image_hdu.data[0,0],origin='lower',
                             levels=levels*sigma,colors='k',linewidths=0.2)
        xlen,ylen = image_hdu.data[0,0].shape
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Declination (J2000)')
        #
        # Adjust limits
        #
        #ax.set_xlim(0.1*xlen,0.9*xlen)
        #ax.set_ylim(0.1*ylen,0.9*ylen)
        #
        # Plot colorbar
        #
        cbar = fig.colorbar(cax,fraction=0.046,pad=0.04)
        cbar.set_label('Flux Density (Jy/beam)')
        #
        # Plot beam
        #
        pixsize = image_hdu.header['CDELT2'] # deg
        if 'BMAJ' in image_hdu.header.keys():
            beam_maj = image_hdu.header['BMAJ']/pixsize # pix
            beam_min = image_hdu.header['BMIN']/pixsize # pix
            beam_pa = image_hdu.header['BPA']
            ellipse = Ellipse((1./6.*xlen,1./6.*ylen),
                            beam_min,beam_maj,angle=beam_pa,
                            fill=True,zorder=10,hatch='///',
                            edgecolor='black',facecolor='white')
            ax.add_patch(ellipse)
        elif len(hdulist) > 1:
            hdu = hdulist[1]
            convert = 1.
            # convert arcsec to deg if necessary
            if 'arcsec' in hdu.header['TUNIT1']:
                convert = 1./3600.
            beam_maj = convert*hdu.data['BMAJ'][0]/pixsize
            beam_min = convert*hdu.data['BMIN'][0]/pixsize
            beam_pa = hdu.data['BPA'][0]
            ellipse = Ellipse((1./8.*xlen,1./8.*ylen),
                            beam_min,beam_maj,angle=beam_pa,
                            fill=True,zorder=10,hatch='///',
                            edgecolor='black',facecolor='white')
            ax.add_patch(ellipse)
        #
        # Plot WISE Catalog regions
        #
        if wisefile is not None:
            wisedata = np.genfromtxt(wisefile,dtype=None,names=True,encoding='UTF-8')
            # limit only to regions with centers within image
            corners = wcs_celest.calc_footprint()
            min_RA = np.min(corners[:,0])
            max_RA = np.max(corners[:,0])
            RA_range = max_RA - min_RA
            min_Dec = np.min(corners[:,1])
            max_Dec = np.max(corners[:,1])
            Dec_range = max_Dec - min_Dec
            good = (min_RA < wisedata['RA'])&(wisedata['RA'] < max_RA)&(min_Dec < wisedata['Dec'])&(wisedata['Dec'] < max_Dec)
            # plot them
            wisedata = wisedata[good]
            for dat in wisedata:
                xpos,ypos = wcs_celest.wcs_world2pix(dat['RA'],dat['Dec'],1)
                diameter = dat['Size']*2./3600./pixsize
                ell = Ellipse((xpos,ypos),diameter,diameter,
                               color='y',fill=False,linestyle='dashed',zorder=100)
                ax.add_patch(ell)
                ax.text(dat['RA'],dat['Dec'],dat['GName'],transform=ax.get_transform('world'),fontsize=2,zorder=100)
            # add legend element
            ell = Ellipse((0,0),0.1,0.1,color='y',fill=False,
                          linestyle='dashed',label='WISE Catalog')
            patches = [ell]
            wise_legend = plt.legend(handles=patches,loc='lower right',fontsize=6,
                                     handler_map={Ellipse: HandlerEllipse()})
            ax.add_artist(wise_legend)
            #
            # Identify peaks if they don't already exist
            #
            if len(peak_regions) == 0:
                #
                # Generate pixel axes
                #
                axx = np.arange(1, image_hdu.header['NAXIS1']+1)
                axy = np.arange(1, image_hdu.header['NAXIS2']+1)
                pixx, pixy = np.meshgrid(axx, axy)
                #
                # Loop over WISE sources within field
                #
                peak_pixs = []
                peak_gnames = []
                peak_seps = []
                for dat in wisedata:
                    #
                    # Get image clipped at levels[0] * sigma
                    #
                    clip = image_hdu.data[0,0] > levels[0]*sigma
                    #
                    # Also clip within WISE region
                    #
                    xpos,ypos = wcs_celest.wcs_world2pix(dat['RA'],dat['Dec'],1)
                    radius = dat['Size']/3600./pixsize
                    clip = clip * ((pixx-xpos)**2. + (pixy-ypos)**2. < radius**2.)
                    #
                    # Skip if no bright emission within WISE region
                    #
                    if np.sum(clip) == 0:
                        continue
                    #
                    # Get location of peak within PB-corrected image
                    #
                    clip_data = pbcorr_hdu.data[0,0] * clip
                    peak_pix = np.nanargmax(clip_data)
                    peak_pix = (pixx.flatten()[peak_pix],
                                pixy.flatten()[peak_pix])
                    #
                    # Get separation between this peak and the
                    # center of the WISE source
                    #
                    peak_sep = np.sqrt(
                        (peak_pix[0]-xpos)**2. + (peak_pix[1]-ypos)**2.)
                    #
                    # If the peak position is on the edge of the WISE
                    # region, then this emission is likely associated
                    # with an overlapping WISE region, so skip it.
                    # use a 1 pixel buffer.
                    #
                    if radius - peak_sep < 1.:
                        continue
                    #
                    # Multiple WISE sources might have the same peak
                    # pix if they overlap. In this case, choose the
                    # smallest wise source.
                    #
                    if peak_pix in peak_pixs:
                        idx = peak_pixs.index(peak_pix)
                        if peak_sep < peak_seps[idx]:
                            peak_seps[idx] = peak_sep
                            peak_gnames[idx] = dat['GName']
                    else:
                        peak_pixs.append(peak_pix)
                        peak_seps.append(peak_sep)
                        peak_gnames.append(dat['GName'])
                #
                # Save region files
                #
                for pix, gname in zip(peak_pixs, peak_gnames):
                    #
                    # Convert pixel to WCS
                    #
                    ra, dec = wcs_celest.wcs_pix2world(pix[0], pix[1], 1)
                    coord = SkyCoord(ra, dec, frame='fk5', unit=('deg', 'deg'))
                    #
                    # Save region file
                    #
                    rgnfname = '{0}{1}'.format(gname, rgnend)
                    with open(rgnfname, 'w') as f:
                        f.write('#CRTFv0 CASA Region Text Format version 0\n')
                        f.write('ellipse [[{0:02d}:{1:02d}:{2:08.5f}, {3:+02d}.{4:02d}.{5:07.4f}], [1.0arcsec, 1.0arcsec], 90.00000000deg] coord=J2000, corr=[I, Q, U, V], linewidth=1, linestyle=-, symsize=1, symthick=1, color=magenta, font="DejaVu Sans", fontsize=11, fontstyle=normal, usetex=false\n'.format(
                            int(coord.ra.hms.h), int(coord.ra.hms.m), coord.ra.hms.s,
                            int(coord.dec.dms.d), int(abs(coord.dec.dms.m)), abs(coord.dec.dms.s)))
                    peak_regions.append(rgnfname)
        #
        # Plot regions
        #
        full_regions = [reg.replace('.rgn','.{0}.fullrgn.fits'.format(spw)) for reg in peak_regions]
        labels = [reg.replace(rgnend,'') for reg in peak_regions]
        cmap_jet = plt.get_cmap('jet')
        colors = iter(cmap_jet(np.linspace(0,1,len(peak_regions))))
        for peak_reg,full_reg,lab in zip(peak_regions,full_regions,labels):
            col = next(colors)
            if os.path.exists(peak_reg):
                # plot peak region as "+"
                with open(peak_reg,'r') as f:
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
                    ax.plot(coord.ra.value,coord.dec.value,'+',color=col,
                            markersize=10,transform=ax.get_transform('world'),
                            label=lab)
            if os.path.exists(full_reg):
                # plot full region(s) as contour
                full_hdu = fits.open(full_reg)[0]
                ax.contour(full_hdu.data[0,0],colors=[col],
                           zorder=110,linewidths=0.5)
        #
        # Add regions legend
        #
        if len(peak_regions) > 0:
            region_legend = plt.legend(loc='upper right',fontsize=6)
            ax.add_artist(region_legend)
        #
        # Re-scale to fit, then save
        #
        fig.savefig(image.replace('.fits','.reg.pdf'),
                    bbox_inches='tight')
        plt.close(fig)
        plt.ion()
        outimages.append(image.replace('.fits','.reg.pdf'))
    #
    # Generate PDF of all images
    #
    outimages = ['{'+fn.replace('.pdf','')+'}.pdf' for fn in outimages]
    fname = '{0}'.format(field)
    if taper: fname += '.uvtaper'
    if imsmooth: fname += '.imsmooth'
    fname += '.regionplots.tex'
    with open(fname,'w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        for i in range(0,len(outimages),6):
            f.write(r"\begin{figure}"+"\n")
            f.write(r"\centering"+"\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+outimages[i]+r"}" + "\n")
            if len(outimages) > i+1: f.write(r"\includegraphics[width=0.45\textwidth]{"+outimages[i+1]+r"} \\" + "\n")
            if len(outimages) > i+2: f.write(r"\includegraphics[width=0.45\textwidth]{"+outimages[i+2]+r"}" + "\n")
            if len(outimages) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+outimages[i+3]+r"} \\" + "\n")
            if len(outimages) > i+4: f.write(r"\includegraphics[width=0.45\textwidth]{"+outimages[i+4]+r"}" + "\n")
            if len(outimages) > i+5: f.write(r"\includegraphics[width=0.45\textwidth]{"+outimages[i+5]+r"} \\" + "\n")
            f.write(r"\end{figure}"+"\n")
            f.write(r"\clearpage"+"\n")
        f.write(r"\end{document}")
    os.system('pdflatex -interaction=batchmode {0}'.format(fname))
    
