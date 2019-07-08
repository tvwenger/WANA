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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

__version__ = "1.0"

def main(field,regions,spws,label,
         fluxtype='peak',taper=False,imsmooth=False,
         skip_plot=False,outfile='cont_info.txt'):
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
        # 0    1           2          3        4        5          6           7           8          9
        # spw  frequency   cont       e_cont_A e_cont_B area_arcsec area_pixel beam_arcsec beam_pixel ps_check
        # #    MHz         mJy/beam   mJy/beam mJy/beam arcsec2     pixels     arcsec2     pixels     
        # cont 9494.152594 12345.6789 123.4567 123.4567 12345.67    123456.78  12345.67    123456.78  100.00
        # 1    9494.152594 12345.6789 123.4567 123.4567 12345.67    123456.78  12345.67    123456.78  100.00
        # 1234 12345678902 1234567890 12345678 12345678 12345678901 1234567890 12345678901 1234567890 12345678
        #
        headerfmt = '{0:4} {1:12} {2:10} {3:8} {4:8} {5:11} {6:10} {7:11} {8:10} {9:8}\n'
        rowfmt = '{0:4} {1:12.6f} {2:10.4f} {3:8.4f} {4:8.4f} {5:11.2f} {6:10.2f} {7:11.2f} {8:10.2f} {9:8.2f}\n'
        f.write(headerfmt.format('spw','frequency','cont','e_cont_A','e_cont_B','area_arcsec','area_pixel','beam_arcsec','beam_pixel','ps_check'))
        if fluxtype == 'total':
            fluxunit = 'mJy'
        else:
            fluxunit = 'mJy/beam'
        f.write(headerfmt.format('#','MHz',fluxunit,fluxunit,fluxunit,'arcsec2','pixels','arcsec2','pixels','%'))
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
            # Construct synthesized beam kernel 2D Gaussian centered on origin
            # N.B. Fits stored as transpose
            #
            y_size, x_size = image_hdu.data[0,0].shape
            x_axis = np.arange(x_size) - x_size/2.
            y_axis = np.arange(y_size) - y_size/2.
            y_grid, x_grid = np.meshgrid(y_axis, x_axis, indexing='ij')
            lon_grid = -image_hdu.header['CDELT1']*x_grid
            lat_grid = image_hdu.header['CDELT2']*y_grid
            bmin = image_hdu.header['BMIN']/(2.*np.sqrt(2.*np.log(2.)))
            bmaj = image_hdu.header['BMAJ']/(2.*np.sqrt(2.*np.log(2.)))
            bpa = -np.deg2rad(image_hdu.header['BPA'])+np.pi/2.
            # 2-D Gaussian parameters
            A = np.cos(bpa)**2./(2.*bmaj**2.) + np.sin(bpa)**2./(2.*bmin**2.)
            B = -np.sin(2.*bpa)/(4.*bmaj**2.) + np.sin(2.*bpa)/(4.*bmin**2.)
            C = np.sin(bpa)**2./(2.*bmaj**2.) + np.cos(bpa)**2./(2.*bmin**2.)
            beam_kernel = np.exp(-(A*lon_grid**2. + 2.*B*lon_grid*lat_grid + C*lat_grid**2.))
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
            image_rms[np.isinf(image_rms)] = np.nan
            #
            # Read region file, extract data from region center pixel
            #
            if not os.path.exists(region):
                cont = np.nan
                e_cont_A = np.nan
                e_cont_B = np.nan
                area_pixel = np.nan
                area_arcsec = np.nan
                ps_check = np.nan
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
                e_cont_A = 1000.*image_rms[int(pix[1]),int(pix[0])]
                e_cont_B = np.nan
                ps_check = np.nan
            #
            # Read region file, sum spectrum region
            #
            else:
                region_mask = np.array(fits.open(region)[0].data[0,0],dtype=np.bool)
                area_pixel = np.sum(region_mask)
                area_arcsec = area_pixel * pixel_size
                #
                # Compute flux, mJy/beam*pixels/beam_area_pixels
                #
                cont = 1000.*np.sum(image_hdu.data[0,0,region_mask])/beam_pixel # mJy
                #
                # Compute primary beam corrected error formally
                # taking into consideration correlated noise by
                # using synthesized beam kernel centered on each pixel
                # N.B. transpose when rolling beam_kernel
                #
                p_sigma = np.array([np.nansum(image_rms * np.roll(beam_kernel, (int(y), int(x)), axis=(0,1)))
                                   for x, y in zip(x_grid[region_mask], y_grid[region_mask])])
                e_cont_A = 1000.*np.sqrt(np.sum(image_rms[region_mask] * p_sigma))/beam_pixel # mJy
                #
                # Compute primary beam corrected error approximation
                # Since noise is spatially correlated on beam scales
                # e_cont ~ mJy/beam * sqrt(region_area/beam_area) 
                #
                e_cont_B = 1000.*np.mean(image_rms[region_mask]) * np.sqrt(area_pixel/beam_pixel) # mJy
                #
                # Extract sub image, FFT and divide by beam kernel
                # to get de-convolved visibilities
                #
                sub_image = image_hdu.data[0,0] * region_mask
                sub_image[np.isnan(sub_image)] = 0.
                sub_vis = np.fft.fftshift(np.fft.fft2(sub_image))
                beam_vis = np.fft.fftshift(np.fft.fft2(beam_kernel))
                beam_vis[np.abs(beam_vis) < 1.e-3] = np.nan
                deconvolve_vis = sub_vis/beam_vis
                #
                # FFT the grid to get uv distances
                #
                u_axis = np.fft.fftshift(np.fft.fftfreq(len(x_axis), d=-np.deg2rad(image_hdu.header['CDELT1'])))
                v_axis = np.fft.fftshift(np.fft.fftfreq(len(y_axis), d=np.deg2rad(image_hdu.header['CDELT2'])))
                v_grid, u_grid = np.meshgrid(v_axis, u_axis, indexing='ij')
                uvwave = np.sqrt(u_grid**2. + v_grid**2.).flatten()
                vis = 1000. * np.abs(deconvolve_vis.flatten()) # mJy
                #
                # Mask large uvwaves
                #
                uvwave_max = 4.*np.log(2.)/(np.pi*np.deg2rad(image_hdu.header['BMAJ']))
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
                ps_check = 100.*(vis_max-vis_zero)/vis_zero
                #
                # Plot visibilities
                #
                if not skip_plot:
                    xmin = 0.
                    xmax = uvwave_max
                    ymin = -0.1*np.nanmax(vis)
                    ymax = 1.1*np.nanmax(vis)
                    plt.ioff()
                    fig, ax = plt.subplots()
                    h = ax.hist2d(uvwave, vis, # mJy
                                  bins=20, norm=LogNorm(),
                                  range=[[xmin,xmax],[ymin,ymax]])
                    cb = fig.colorbar(h[3])
                    cb.set_label("Number of Pixels")
                    ax.set_xlabel(r"{\it uv} distance (wavelengths)")
                    ax.set_ylabel(r"Amplitude (mJy)")
                    title = '{0}.{1}'.format(label,spw)
                    ax.set_title(title)
                    fig.tight_layout()
                    fig.savefig('{0}.{1}.vis.pdf'.format(label,spw))
                    plt.close(fig)
                    plt.ion()
            #
            # Write row to file
            #
            f.write(rowfmt.format(spw.replace('spw',''),frequency,cont,e_cont_A,e_cont_B,area_arcsec,area_pixel,beam_arcsec,beam_pixel,ps_check))
