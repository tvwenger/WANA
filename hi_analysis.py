"""
hi_analysis.py - WANA HI analysis program

Analyze spectral data cubes, plot HI spectra, measure optical depth.

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
Trey V. Wenger December 2019 - V1.0
"""

import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import shutil
import itertools

__version__ = "1.0"

class ClickPlot:
    """
    Generic class for generating and interacting with matplotlib 
    figures
    """
    def __init__(self, num, title):
        """
        Initialize a new interactive matplotlib figure

        Inputs: num
          num :: integer
            The matplotlib figure number
        
        Returns: Nothing
        """
        self.fig = plt.figure(num,figsize=(12,4))
        plt.clf()
        self.ax = self.fig.add_subplot(111)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []
        self.title = title

    def onclick(self,event):
        """
        Handle click event

        Inputs: event
          event :: matplotlib event
            The received event

        Returns: Nothing
        """
        # check that event is left or right mouse click
        if event.button not in [1,3]:
            self.clickbutton.append(-1)
            return
        # check if click is within plot axes
        if not event.inaxes:
            self.clickbutton.append(-1)
            return
        self.clickbutton.append(event.button)
        self.clickx_data.append(event.xdata)
        self.clicky_data.append(event.ydata)

    def line_free(self, xdata, ydata, rms, xlabel=None,ylabel=None):
        """
        Using click events to get the line free regions of a spectrum

        Inputs: 
          xdata :: 1-D array of scalars
            The x-axis data for the plot
          ydata :: 1-D array of scalars
            The y-axis data for the plot
          rms :: 1-D array of scalars
            The rms at each xdata
          xlabel :: string
            The x-axis label
          ylabel :: string
            The y-axis label

        Returns: regions
          regions :: list of list of scalars
            Each row is a list of a 2 scalar list defining the
            start and end xdata value of a line-free region
            [[start0,end0], [start1,end1], ...]
        """
        #
        # set up the figure
        self.ax.clear()
        self.ax.grid(False)
        self.ax.plot(xdata,ydata,'k-')
        self.ax.plot(xdata, ydata+1.0*rms, 'k-', alpha=1.0, linewidth=0.1)
        self.ax.plot(xdata, ydata-1.0*rms, 'k-', alpha=1.0, linewidth=0.1)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(np.min(xdata),np.max(xdata))
        yrange = np.max(ydata+1.0*rms)-np.min(ydata-1.0*rms)
        ymin = np.min(ydata-1.0*rms)-0.10*yrange
        ymax = np.max(ydata+1.0*rms)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.ax.set_title(self.title)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []
        #
        # get user clicks
        #
        print("Left click to select start of line-free-region.")
        print("Left click again to select end of line-free-region.")
        print("Repeat as necessary.")
        print("Right click when done.")
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.onclick)
        self.fig.tight_layout()
        self.fig.show()
        nregions = []
        while True:
            self.fig.waitforbuttonpress()
            if self.clickbutton[-1] == 3:
                if len(nregions) == 0 or len(nregions) % 2 != 0:
                    continue
                else:
                    break
            elif self.clickbutton[-1] == 1:
                if self.clickx_data[-1] < np.min(xdata):
                    nregions.append(np.min(xdata))
                elif self.clickx_data[-1] > np.max(xdata):
                    nregions.append(np.max(xdata))
                else:
                    nregions.append(self.clickx_data[-1])
                self.ax.axvline(nregions[-1])
                self.fig.show()
        self.fig.canvas.mpl_disconnect(cid)
        #
        # zip regions into 2-d list
        #
        regions = zip(nregions[::2],nregions[1::2])
        return regions

    def plot_contfit(self,xdata, ydata, rms, contfit,
                     xlabel=None, ylabel=None):
        """
        Plot data and continuum fit

        Inputs: 
          xdata :: 1-D array of scalars
            The x-axis data for the plot
          ydata :: 1-D array of scalars
            The y-axis data for the plot
          rms :: 1-D array of scalars
            The rms at each xdata
          contfit :: 1-D array of scalars
            The continuum fit at each xdata point
          xlabel :: string
            The x-axis label
          ylabel :: string
            The y-axis label

        Returns: Nothing
        """
        #
        # set-up figure
        #
        self.ax.clear()
        self.ax.grid(False)
        self.ax.plot(xdata,ydata,'k-')
        self.ax.plot(xdata, ydata+1.0*rms, 'k-', alpha=1.0, linewidth=0.1)
        self.ax.plot(xdata, ydata-1.0*rms, 'k-', alpha=1.0, linewidth=0.1)
        self.ax.plot(xdata,contfit(xdata),'r-')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(np.min(xdata),np.max(xdata))
        yrange = np.max(ydata+1.0*rms)-np.min(ydata-1.0*rms)
        ymin = np.min(ydata-1.0*rms)-0.10*yrange
        ymax = np.max(ydata+1.0*rms)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        self.fig.show()
        print("Click anywhere to continue")
        self.fig.waitforbuttonpress()

    def plot_tau(self,xdata,ydata,rms,
                 xlabel=None,ylabel=None,outfile=None,
                 vlsr=None, e_vlsr=None, vlsr_tan=None,
                 e_vlsr_tan_neg=None, e_vlsr_tan_pos=None,
                 start=None, end=None,
                 cont=None, e_cont=None, cont_unit=None,
                 ew=None, e_ew=None, tau_rms=None):
        """
        Plot optical depth spectrum. Save figure to file.

        Inputs:
          xdata :: 1-D array of scalars
            The plot x-data.
          ydata :: 1-D array of scalars
            The plot y-data.
          rms :: 1-D array of scalars
            The rms at each x-data
          xlabel :: string
            The x-axis label.
          ylabel :: string
            The y-axis label.
          outfile :: string
            If not None, the filename where to save this figure
          vlsr, e_vlsr :: scalar
            The HII region LSR velocity and uncertainty (km/s)
          vlsr_tan, e_vlsr_tan_neg, e_vlsr_tan_pos :: scalar
            The tangent point LSR velocity and uncertainty (km/s)
          start, end :: integer
            The indicies of xdata over which the optical depth is calculated
          cont, e_cont :: scalar
            The continuum brightness or flux and uncertainty
          cont_unit :: string
            The continuum brightness unit
          ew, e_ew :: scalar
            The equvialent width and uncertainty
          tau_rms :: scalar
            The mean optical depth rms

        Returns: Nothing
        """
        self.ax.clear()
        self.ax.grid(True)
        self.ax.axhline(1, color='k')
        #
        # Plot data
        #
        self.ax.plot(xdata,ydata,'k-')
        self.ax.plot(xdata, 1.+1.0*rms, 'k-', alpha=1.0, linewidth=0.1)
        self.ax.plot(xdata, 1.-1.0*rms, 'k-', alpha=1.0, linewidth=0.1)
        #
        # Add plot labels
        #
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        xmin = -150.
        xmax = 100.
        self.ax.set_xlim(xmin,xmax)
        yrange = np.max(ydata)-np.min(ydata)
        ymin = np.min(ydata)-0.10*yrange
        ymax = np.max(ydata)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.ax.set_title(self.title)
        #
        # Plot peripherals
        #
        if vlsr is not None and vlsr_tan is not None:
            self.ax.axvline(vlsr, linestyle='solid', linewidth=1.5, color='k')
            self.ax.axvspan(vlsr-e_vlsr, vlsr+e_vlsr, alpha=0.5, color='k')
            self.ax.axvline(vlsr_tan, linestyle='dashed', linewidth=1.5, color='k')
            self.ax.axvspan(vlsr_tan-e_vlsr_tan_neg, vlsr_tan+e_vlsr_tan_pos, alpha=0.5, color='k')
        if start is not None and end is not None:
            self.ax.plot([xdata[start],xdata[end+1]], [ymin+0.1*yrange, ymin+0.1*yrange],
                         'k-', linewidth=5.0)
        if cont is not None and ew is not None:
            label = (
                '{0:.1f} '.format(cont)+r'$\pm$'+' {0:.1f} '.format(e_cont)+cont_unit+"\n"+
                r'$\langle\sigma_\tau\rangle$ = '+'{0:.2f}'.format(tau_rms)+"\n"+
                'EW = {0:.1f} '.format(ew)+r'$\pm$'+' {0:.1f} '.format(e_ew)+r'km s$^{-1}$')
            self.ax.text(25., ymin+0.3*yrange, label,
                         bbox=dict(facecolor='white', alpha=1.0))
        elif cont is not None:
            label = (
                '{0:.1f} '.format(cont)+r'$\pm$'+' {0:.1f} '.format(e_cont)+cont_unit+"\n"+
                r'$\langle\sigma_\tau\rangle$ = '+'{0:.2f}'.format(tau_rms))
            self.ax.text(25., ymin+0.3*yrange, label,
                         bbox=dict(facecolor='white', alpha=1.0))
            
        self.fig.tight_layout()
        self.fig.savefig(outfile)
        self.fig.show()
        print("Click anywhere to continue")
        self.fig.waitforbuttonpress()

def dump_spec(imagename, residualname, pbimagename,
              region, fluxtype, smooth_channels):
    """
    Extract spectrum from region.
    
    Inputs: 
      imagename :: string
        FITS image to analyze
      residualname :: string
        Residual FITS image
      pbimagename :: string
        Primary beam FITS image
      region :: string
        Region file to use for spectral extraction
      fluxtype :: string
        'total' to measure integrated flux, 'peak' to measure peak
        flux
      smooth_channels :: scalar
        If > 1.0, smooth the image cube by a Gaussian of this FWHM

    Returns: specdata
      specdata :: ndarray
        array with columns 'channel', 'velocity', and 'flux'
    """
    #
    # Where the spec data is saved
    #
    logfile = '{0}.{1}.specflux'.format(imagename,region)
    print("Extracting spectrum from {0}".format(imagename))
    print("Dumping spectrum to {0}".format(logfile))
    #
    # Open image, get beam area
    #
    hdu = fits.open(imagename)
    res_hdu = fits.open(residualname)[0]
    pb_hdu = fits.open(pbimagename)[0]
    if len(hdu) > 1:
        # need to parse beam table
        image_hdu, beam_hdu = hdu
        #
        # Calculate beam area and pixel size
        # N.B. beamsize is stored in arcsec in beam table
        #
        pixel_size = 3600.**2. * np.abs(image_hdu.header['CDELT1'] * image_hdu.header['CDELT2']) # arcsec^2
        beam_arcsec = np.pi*beam_hdu.data['BMIN']*beam_hdu.data['BMAJ']/(4.*np.log(2.)) # arcsec^2
        beam_pixel = beam_arcsec / pixel_size
    else:
        # no beam table
        image_hdu = hdu[0]
        #
        # Calculate beam area and pixel size
        # N.B. beamsize is stored in deg in image header
        #
        pixel_size = 3600.**2. * np.abs(image_hdu.header['CDELT1'] * image_hdu.header['CDELT2']) # arcsec^2
        beam_arcsec = 3600.**2. * np.pi*image_hdu.header['BMIN']*image_hdu.header['BMAJ']/(4.*np.log(2.)) # arcsec^2
        beam_pixel = beam_arcsec / pixel_size
    #
    # Smooth image data
    #
    if smooth_channels > 1.:
        sigma = smooth_channels / (2.*np.sqrt(2.*np.log(2.)))
        image_data = gaussian_filter(image_hdu.data, sigma=(0., sigma, 0., 0.))
        res_data = gaussian_filter(res_hdu.data, sigma=(0., sigma, 0., 0.))
    else:
        image_data = image_hdu.data
        res_data = res_hdu.data
    #
    # Get WCS, channel and velocity axes
    #
    image_wcs = WCS(image_hdu.header)
    wcs_celest = image_wcs.sub(['celestial'])
    channel = np.arange(image_data.shape[1])
    velocity = ((channel-(image_hdu.header['CRPIX3']-1))*image_hdu.header['CDELT3'] + image_hdu.header['CRVAL3'])/1000. # km/s
    #
    # Compute residual spatial rms in each channel using MAD
    #
    spatial_rms = np.array([1.4826*np.median(np.abs(res_data[0,i,:,:]-np.median(res_data[0,i,:,:])))
                            for i in range(len(channel))]) * 1000. # mJy/beams
    #
    # Read region file, extract spectrum from region center pixel
    #
    if not os.path.exists(region):
        # no region
        return None
    elif fluxtype == 'peak':
        with open(region,'r') as f:
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
        #
        # Get pixel location of coord, extract spectrum
        # N.B. Need to flip vertices to match WCS
        #
        pix = wcs_celest.wcs_world2pix(coord.ra.deg,coord.dec.deg,1)
        spec = image_data[0,:,int(pix[1]),int(pix[0])]*1000. # mJy/beam
        spec[spec == 0.] = np.nan
        #
        # Correct spatial rms for primary beam
        #
        spatial_rms /= pb_hdu.data[0,0,int(pix[1]),int(pix[0])]
    #
    # Read region file, sum spectrum
    #
    else:
        region_mask = np.array(fits.open(region)[0].data[0,0],dtype=np.bool)
        cubedata = np.array([chandata*region_mask
                             for chandata in image_data[0]]) # Jy/beam
        cubedata[cubedata == 0.] = np.nan
        #
        # Sum spectrum
        #
        spec_nowt = np.nansum(cubedata,axis=(1,2))
        spec_nowt = 1000.* spec_nowt / beam_pixel # mJy
        spec_nowt[spec_nowt == 0.] = np.nan
        #
        # Compute weights as median continuum level in each
        # pixel from -200 to -150 and 100 to 150 km/s
        #
        velo_cut = (velocity < -150.)+(velocity > 100.)
        weights = np.nanmedian(cubedata[velo_cut,:,:], axis=0)
        #
        # Computed weighted sum in each pixel
        #
        spec_wt = np.array([np.nansum(chandata * weights)
                            for chandata in cubedata])
        spec_wt = 1000.* spec_wt / beam_pixel # mJy
        spec_wt[spec_wt == 0.] = np.nan
        # re-normalize to continuum level
        spec = spec_wt * np.nanmedian(spec_nowt[velo_cut])/np.nanmedian(spec_wt[velo_cut])
        #
        # Convert spatial_rms to flux using weights
        #
        spatial_rms = np.array([np.nansum(rms*weights)
                                for rms in spatial_rms])/beam_pixel # mJy
        spatial_rms *= np.nanmedian(spec_nowt[velo_cut])/np.nanmedian(spec_wt[velo_cut])
        #
        # Correct rms for primary beam
        #
        spatial_rms /= np.mean(pb_hdu.data[0,0,region_mask])
    #
    # Save spectrum to file
    #
    logfile = '{0}.{1}.specflux'.format(imagename,region)
    with open(logfile,'w') as f:
        f.write('channel velocity flux     rms \n')
        if fluxtype == 'peak':
            f.write('#       km/s     mJy/beam mJy/beam\n')
        else:
            f.write('#       km/s     mJy      mJy/beam\n')
        for chan,vel,sp,rms in zip(channel,velocity,spec,spatial_rms):
            f.write('{0:7} {1:8.2f} {2:8.4f} {3:8.4f}\n'.format(chan,vel,sp,rms))
    #
    # Import spectrum
    #
    specdata = np.genfromtxt(logfile,comments='#',dtype=None,names=True)
    isnan = (specdata['flux'] == 0.) | (np.isinf(specdata['flux']))
    specdata['flux'][isnan] = np.nan
    if np.all(np.isnan(specdata['flux'])):
        # region is outside of primary beam
        return None
    else:
        return specdata
    
def hi_optical_depth(fluxtype, specdata, outfile, title,
                     vlsr=None, e_vlsr=None, vlsr_tan=None,
                     e_vlsr_tan_neg=None, e_vlsr_tan_pos=None):
    """
    Fit continuum baseline, compute optical depth, save spectrum.
    Calculate equivalent width between source velocity and tangent
    point.

    Inputs:
      fluxtype :: string
        'total' to measure total flux, 'peak' to measure peak flux
      specdata :: ndarray
        output from dump_spec()
      outfile :: string
        where to save spectrum figure
      vlsr, e_vlsr :: scalar
        The HII region LSR velocity and uncertainty (km/s)
      vlsr_tan, e_vlsr_tan_neg, e_vlsr_tan_pos :: scalar
        The tangent point LSR velocity and uncertainty (km/s)

    Returns: cont, rms, ew, e_ew, tau_rms
      cont :: scalar
        The background continuum brightness or flux (mJy/beam or mJy)
      rms :: scalar
        Spectral rms (mJy/beam or mJy)
      ew, e_ew :: scalar
        The HI absorption equivalent width and uncertainty (km/s)
      tau_rms :: scalar
        The mean optical depth rms
    """
    #
    # Make the plot
    #
    myplot = ClickPlot(1, title)
    if fluxtype=='peak':
        ylabel = r'Flux Density (mJy beam$^{-1}$)'
        cont_unit = r'mJy beam$^{-1}$'
    else:
        ylabel = 'Flux Density (mJy)'
        cont_unit = 'mJy'
    #
    # Remove NaNs (if any)
    #
    is_nan = np.isnan(specdata['flux'])
    specdata_flux = specdata['flux'][~is_nan]
    specdata_velocity = specdata['velocity'][~is_nan]
    specdata_rms = specdata['rms'][~is_nan]
    #
    # Get line-free regions
    #
    regions = myplot.line_free(specdata_velocity, specdata_flux,
                               specdata_rms, xlabel=r'$V_{\rm LSR}$ (km s$^{-1}$)',
                               ylabel=ylabel)
    #
    # Extract line free velocity and flux
    #
    line_free_mask = np.zeros(specdata_velocity.size,dtype=bool)
    for reg in regions:
        line_free_mask[(specdata_velocity>reg[0])&(specdata_velocity<reg[1])] = True
    line_free_velocity = specdata_velocity[line_free_mask]
    line_free_flux = specdata_flux[line_free_mask]
    line_free_rms = specdata_rms[line_free_mask]
    #
    # Fit baseline as polyonimal order 3
    #
    print("Fitting continuum baseline polynomial...")
    pfit = np.polyfit(line_free_velocity, line_free_flux, 3, w=1./line_free_rms)
    contfit = np.poly1d(pfit)
    myplot.plot_contfit(specdata_velocity, specdata_flux,
                        specdata_rms, contfit,
                        xlabel=r'$V_{\rm LSR}$ (km s$^{-1}$)', ylabel=ylabel)
    print("Done.")
    #
    # Calculate average continuum
    #
    cont_brightness = np.mean(contfit(specdata_velocity))
    cont_brightness_err = np.std(line_free_flux - contfit(line_free_velocity))
    #
    # Compute e^-tau
    #
    exp_tau = specdata_flux/contfit(specdata_velocity)
    #exp_tau_err = np.sqrt(exp_tau**2.*((specdata_rms/specdata_flux)**2. + (cont_brightness_err/cont_brightness)**2.))
    exp_tau_err = np.sqrt(exp_tau**2.*((specdata_rms/specdata_flux)**2.))
    tau_rms = np.mean(np.log(1.+exp_tau_err))
    #
    # Calculate equivalent width between source velocity and tangent
    # point velocity. Start at 15 km/s beyond source velocity,
    # and end 15 km/s beyond tangent point velocity.
    #
    if vlsr is None or vlsr_tan is None:
        start = None
        end = None
        ew = None
        e_ew = None
    elif vlsr_tan > 0.:
        # first quadrant
        # range is vlsr+15. to vlsr_tan+15.
        start = np.argmin(np.abs(specdata_velocity - (vlsr+15.)))
        end = np.argmin(np.abs(specdata_velocity - (vlsr_tan+15.)))
        vlsr_channel = specdata_velocity[1]-specdata_velocity[0]
        ew = np.sum(1.-exp_tau[start:end+1])*vlsr_channel
        e_ew = np.sqrt(np.sum(exp_tau_err**2.))*vlsr_channel
    else:
        # fourth quadrant
        # range is vlsr_tan-15. to vlsr-e15.
        start = np.argmin(np.abs(specdata_velocity - (vlsr_tan-15.)))
        end = np.argmin(np.abs(specdata_velocity - (vlsr-15.)))
        vlsr_channel = specdata_velocity[1]-specdata_velocity[0]
        ew = np.sum(1.-exp_tau[start:end+1])*vlsr_channel
        e_ew = np.sqrt(np.sum(exp_tau_err**2.))*vlsr_channel        
    #
    # Plot optical depth
    #
    myplot.plot_tau(specdata_velocity, exp_tau, exp_tau_err,
                    xlabel=r'$V_{\rm LSR}$ (km s$^{-1}$)', ylabel=r'$e^{-\tau}$',
                    outfile=outfile, vlsr=vlsr, e_vlsr=e_vlsr,
                    vlsr_tan=vlsr_tan, e_vlsr_tan_neg=e_vlsr_tan_neg,
                    e_vlsr_tan_pos=e_vlsr_tan_pos,
                    start=start, end=end,
                    cont=cont_brightness, e_cont=cont_brightness_err,
                    cont_unit=cont_unit,
                    ew=ew, e_ew=e_ew, tau_rms=tau_rms)
    return cont_brightness, cont_brightness_err, ew, e_ew, tau_rms

def main(field,region,spw,stokes='I',
         fluxtype='peak',clean=False,taper=False,imsmooth=False,
         outfile='line_info.txt', smooth_channels=1.0,
         wisefile=None):
    """
    Extract spectrum from region in each data cube, measure continuum
    brightess and compute HI optical depth. Also plot maser spectra.

    Inputs:
      field :: string
        The field name
      region :: string
        The region file used to analyze
      spw :: string
        spw to analyze
      stokes :: string
        The Stokes parameters saved in the images
      fluxtype :: string
        What type of flux to measure. 'peak' to use peak regions and
        measure peak flux density, 'total' to use full regions and
        measure total flux density.
      clean :: string
        If True, use clean image. Otherwise, use dirty.
      taper :: boolean
        if True, use uv-tapered images
      imsmooth :: boolean
        if True, use imsmooth images
      outfile :: string
        Filename where the output table is written
      smooth_channels :: scalar
        Smooth the image cube with a Gaussian of this FWHM
      wisefile :: string
        Location of WISE catalog detections file

    Returns: Nothing
    """
    #
    # Check cube exists
    #
    pbimagename = '{0}.spw{1}.{2}.channel.pb.fits'.format(field,spw,stokes)
    imagename = '{0}.spw{1}.{2}.channel'.format(field,spw,stokes)
    if clean:
        imagename += '.clean'
    else:
        imagename += '.dirty'
    if taper: imagename += '.uvtaper'
    if imsmooth: imagename += '.imsmooth'
    residualname = imagename+'.residual.fits'
    imagename += '.pbcor.image.fits'
    if not os.path.exists(imagename):
        print("{0} not found.".format(imagename))
        return
    if not os.path.exists(residualname):
        print("{0} not found.".format(residualname))
        return
    if not os.path.exists(pbimagename):
        print("{0} not found.".format(pbimagename))
        return
    #
    # extract spectrum and 
    #
    specdata = dump_spec(imagename, residualname, pbimagename,
                         region, fluxtype, smooth_channels)
    if specdata is None:
        return
    #
    # Read HII region detections and match based on GName
    #
    gname = region[0:15]
    detections = np.genfromtxt(wisefile, dtype=None, names=True,
                               encoding='utf-8')
    good = detections['GName'] == gname
    if np.sum(good) == 0:
        print("{0} not found in detections file!".format(gname))
        vlsr = None
        e_vlsr = None
        vlsr_tan = None
        e_vlsr_tan_neg = None
        e_vlsr_tan_pos = None
    else:
        vlsr = detections['VLSR'][good][0]
        e_vlsr = detections['e_VLSR'][good][0]
        vlsr_tan = detections['Vtan'][good][0]
        e_vlsr_tan_neg = detections['e_Vtan_neg'][good][0]
        e_vlsr_tan_pos = detections['e_Vtan_pos'][good][0]
    #
    # Compute and plot HI optical depth
    #
    fname = '{0}.HI.{1}.channel'.format(region,stokes)
    if clean:
        fname += '.clean'
    else:
        fname += '.dirty'
    if taper: fname += '.uvtaper'
    if imsmooth: fname += '.imsmooth'
    fname += '.spec.pdf'
    title = '{0} '.format(gname)
    if clean:
        title += '({0}/clean)'.format(fluxtype)
    else:
        title += '({0}/dirty)'.format(fluxtype)
    cont, rms, ew, e_ew, tau_rms = hi_optical_depth(
        fluxtype, specdata, fname, title, vlsr=vlsr, e_vlsr=e_vlsr,
        vlsr_tan=vlsr_tan, e_vlsr_tan_neg=e_vlsr_tan_neg,
        e_vlsr_tan_pos=e_vlsr_tan_pos)
    #
    # Set-up file
    #
    with open(outfile,'w') as f:
        # 0       1          2          3       4       5 
        # lineid  cont       rms        EW      EW_err  tau_rms
        # #       mJy/(beam) mJy/(beam)
        # HI      1000000.00 1000000.00 1000.00 1000.00 1000.00
        # 1234567 1234567890 1234567890 1234567 1234567 1234567
        #
        headerfmt = '{0:7} {1:10} {2:10} {3:7} {4:7} {5:7}\n'
        rowfmt = '{0:7} {1:10.2f} {2:10.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n'
        f.write(headerfmt.format('lineid','cont','rms','EW','EW_err','tau_rms'))
        if fluxtype == 'total':
            fluxunit = 'mJy'
        else:
            fluxunit = 'mJy/beam'
        f.write(headerfmt.format('#',fluxunit,fluxunit,'','',''))
        if ew is None:
            f.write(rowfmt.format('HI',cont,rms,np.nan,np.nan,tau_rms))
        else:
            f.write(rowfmt.format('HI',cont,rms,ew,e_ew,tau_rms))
