"""
line_analysis.py - WANA spectral line analysis program

Analyze spectral data cubes, fit Gaussians, measure line properties,
etc.

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
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import logging
import logging.config
import ConfigParser
import shutil
import itertools

__version__ = "1.0"

# load logging configuration file
logging.config.fileConfig('logging.conf')

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

class ClickPlot:
    """
    Generic class for generating and interacting with matplotlib 
    figures
    """
    def __init__(self,num):
        """
        Initialize a new interactive matplotlib figure

        Inputs: num
          num :: integer
            The matplotlib figure number
        
        Returns: Nothing
        """
        self.fig = plt.figure(num,figsize=(8,6))
        plt.clf()
        self.ax = self.fig.add_subplot(111)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []

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

    def line_free(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Using click events to get the line free regions of a spectrum

        Inputs: xdata, ydata, xlabel, ylabel, title
          xdata :: 1-D array of scalars
            The x-axis data for the plot
          ydata :: 1-D array of scalars
            The y-axis data for the plot
          xlabel :: string
            The x-axis label
          ylabel :: string
            The y-axis label
          title :: string
            The plot title

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
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(np.min(xdata),np.max(xdata))
        yrange = np.max(ydata)-np.min(ydata)
        ymin = np.min(ydata)-0.10*yrange
        ymax = np.max(ydata)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
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

    def auto_line_free(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Automatically get the line free regions of a spectrum

        Inputs: xdata, ydata, xlabel, ylabel, title
          xdata :: 1-D array of scalars
            The x-axis data for the plot
          ydata :: 1-D array of scalars
            The y-axis data for the plot
          xlabel :: string
            The x-axis label
          ylabel :: string
            The y-axis label
          title :: string
            The plot title

        Returns: regions
          regions :: list of list of scalars
            Each row is a list of a 2 scalar list defining the
            start and end xdata value of a line-free region
            [[start0,end0], [start1,end1], ...]
        """
        #
        # set-up the figure
        #
        self.ax.clear()
        self.ax.grid(False)
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(np.min(xdata),np.max(xdata))
        yrange = np.max(ydata)-np.min(ydata)
        ymin = np.min(ydata)-0.10*yrange
        ymax = np.max(ydata)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.fig.tight_layout()
        self.fig.show()
        #
        # Iterate fitting a 3rd order polynomial baseline and
        # rejecting outliers until convergence
        # do this on data smoothed by gaussian 3 channels
        #
        smoy = gaussian_filter(ydata,sigma=5.)
        self.ax.plot(xdata,smoy,'g-')
        self.fig.show()
        outliers = np.isnan(ydata)
        while True:
            pfit = np.polyfit(xdata[~outliers],smoy[~outliers],3)
            yfit = np.poly1d(pfit)
            new_smoy = smoy - yfit(xdata)
            rms = 1.4826*np.median(np.abs(new_smoy[~outliers]-np.mean(new_smoy[~outliers])))
            new_outliers = (np.abs(new_smoy) > 3.*rms) | np.isnan(ydata)
            if np.sum(new_outliers) <= np.sum(outliers):
                break
            outliers = new_outliers
        #
        # line-free regions are all channels without outliers
        #
        regions = []
        chans = range(len(xdata))
        for val,ch in itertools.groupby(chans,lambda x: outliers[x]):
            if not val: # if not an outlier
                chs = list(ch)
                regions.append([xdata[chs[0]],xdata[chs[-1]]])
                self.ax.axvline(xdata[chs[0]])
                self.ax.axvline(xdata[chs[-1]])
                self.fig.show()
        return regions

    def plot_contfit(self,xdata,ydata,contfit,
                     xlabel=None,ylabel=None,title=None,auto=False):
        """
        Plot data and continuum fit

        Inputs: xdata, ydata, contfit, xlabel, ylabel, title, auto
          xdata :: 1-D array of scalars
            The x-axis data for the plot
          ydata :: 1-D array of scalars
            The y-axis data for the plot
          contfit :: 1-D array of scalars
            The continuum fit at each xdata point
          xlabel :: string
            The x-axis label
          ylabel :: string
            The y-axis label
          title :: string
            The plot title
          auto :: boolean
            If False, wait for user to review fit

        Returns: Nothing
        """
        #
        # set-up figure
        #
        self.ax.clear()
        self.ax.grid(False)
        self.ax.plot(xdata,contfit(xdata),'r-')
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(np.min(xdata),np.max(xdata))
        yrange = np.max(ydata)-np.min(ydata)
        ymin = np.min(ydata)-0.10*yrange
        ymax = np.max(ydata)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.fig.tight_layout()
        self.fig.show()
        #
        # If not batch-mode, wait for user to review
        #
        if not auto:
            print("Click anywhere to continue")
            self.fig.waitforbuttonpress()

    def get_gauss(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Using click events to get the gaussian fit estimates

        Inputs: xdata, ydata, xlabel, ylabel, title
          xdata :: 1-D array of scalars
            The x-axis data for the plot
          ydata :: 1-D array of scalars
            The y-axis data for the plot
          xlabel :: string
            The x-axis label
          ylabel :: string
            The y-axis label
          title :: string
            The plot title

        Returns: line_start, center_guesses, sigma_guesses, line_end
          line_start :: scalar
            The x-data position of the start of the line region
          center_guesses :: 1-D array of scalars
            The x-data positions of the centers of the lines. One
            element for each Gaussian component.
          sigma_guesses :: 1-D array of scalars
            The estimates of the line widths in x-data units. One
            element for each Gaussian component
          line_end :: scalar
            The x-data position of the end of the line region
        """
        #
        # set-up the figure
        #
        self.ax.clear()
        self.ax.grid(False)
        self.ax.axhline(0,color='k')
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        xmin = -250
        xmax = 150
        self.ax.set_xlim(xmin,xmax)
        ydata_cut = ydata[np.argmin(np.abs(xdata-xmin)):np.argmin(np.abs(xdata-xmax))]
        yrange = np.max(ydata_cut)-np.min(ydata_cut)
        ymin = np.min(ydata_cut)-0.10*yrange
        ymax = np.max(ydata_cut)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []
        #
        # User interactively gets line estimates or skips fitting
        #
        print "Right click to skip fitting this line, or:"
        print "Left click to select start of line region"
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.onclick)
        self.fig.tight_layout()
        self.fig.show()
        self.fig.waitforbuttonpress()
        if 3 in self.clickbutton:
            return None,None,None,None
        self.ax.axvline(self.clickx_data[-1])
        self.fig.show()
        line_start = self.clickx_data[-1]
        guesses = []
        #
        # Get line estimates
        #
        print "Left click to select center of line"
        print "then left click to select width of line."
        print "Repeat for each line, then"
        print "left click to select end of line region."
        print "Right click when finished."
        while True:
            self.fig.waitforbuttonpress()
            if self.clickbutton[-1] == 3:
                break
            self.ax.axvline(self.clickx_data[-1])
            self.fig.show()
            guesses.append(self.clickx_data[-1])
        #
        # Save estimates as arrays
        #
        center_guesses = np.array(guesses[0:-1:2])
        sigma_guesses = np.array(guesses[1:-1:2])-center_guesses
        line_end = guesses[-1]
        self.fig.canvas.mpl_disconnect(cid)
        return line_start,center_guesses,sigma_guesses,line_end

    def auto_get_gauss(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Automatically get the gaussian fit estimates. Only fits
        a single Gaussian component.

        Inputs: xdata, ydata, xlabel, ylabel, title
          xdata :: 1-D array of scalars
            The x-axis data for the plot
          ydata :: 1-D array of scalars
            The y-axis data for the plot
          xlabel :: string
            The x-axis label
          ylabel :: string
            The y-axis label
          title :: string
            The plot title

        Returns: line_start, center_guesses, sigma_guesses, line_end
          line_start :: scalar
            The x-data position of the start of the line region
          center_guesses :: 1-D array of scalars
            The x-data positions of the centers of the lines. Only
            one element.
          sigma_guesses :: 1-D array of scalars
            The estimates of the line widths in x-data units. Only
            one element.
          line_end :: scalar
            The x-data position of the end of the line region
        """
        #
        # set-up figure
        #
        self.ax.clear()
        self.ax.grid(False)
        self.ax.axhline(0,color='k')
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        xmin = -250
        xmax = 150
        self.ax.set_xlim(xmin,xmax)
        ydata_cut = ydata[np.argmin(np.abs(xdata-xmin)):np.argmin(np.abs(xdata-xmax))]
        yrange = np.max(ydata_cut)-np.min(ydata_cut)
        ymin = np.min(ydata_cut)-0.10*yrange
        ymax = np.max(ydata_cut)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.fig.tight_layout()
        self.fig.show()
        #
        # Iterate rejecting outliers until convergence
        # do this on data smoothed by gaussian 3 channels
        #
        smoy = gaussian_filter(ydata,sigma=3.)
        self.ax.plot(xdata,smoy,'g-')
        self.fig.show()
        outliers = np.isnan(ydata)
        while True:
            rms = 1.4826*np.median(np.abs(smoy[~outliers]-np.mean(smoy[~outliers])))
            new_outliers = (np.abs(smoy) > 5.*rms) | np.isnan(ydata)
            if np.sum(new_outliers) <= np.sum(outliers):
                break
            outliers = new_outliers
        #
        # Group outlier regions where ydata values are positive
        # and keep the widest region
        #
        line = np.array([])
        chans = range(len(xdata))
        for val,ch in itertools.groupby(chans,lambda x: outliers[x]):
            if val: # if an outlier
                chs = np.array(list(ch))
                # skip if outliers are negative
                if np.sum(ydata[chs]) < 0.:
                    continue
                # skip if this region is smaller than 4 channels
                if len(chs) < 4:
                    continue
                # skip if fewer than 4 (smoothed) values in this region > 5 rms
                if np.sum(smoy[chs] > 5.*rms) < 4:
                    continue
                # skip if this region is smaller than the saved region
                if len(chs) < len(line):
                    continue
                line = xdata[chs]
        # no line to fit if line is empty
        if len(line) == 0:
            #self.fig.waitforbuttonpress()
            return None,None,None,None
        line_start = np.min(line)-10.
        self.ax.axvline(line_start)
        line_center = np.mean(line)
        self.ax.axvline(line_center)
        line_end = np.max(line)+10.
        self.ax.axvline(line_end)
        line_width = (line_end-line_center)/2.
        self.ax.axvline(line_center+line_width)
        self.fig.show()
        #plt.pause(0.1)
        #self.fig.waitforbuttonpress()
        return line_start,[line_center],[line_width],line_end

    def plot_fit(self,xdata,ydata,line_start,line_end,amp,center,sigma,
                 xlabel=None,ylabel=None,title=None,outfile=None,
                 auto=False):
        """
        Plot data, fit, and residuals. Save figure to file.

        Inputs:
          xdata :: 1-D array of scalars
            The plot x-data.
          ydata :: 1-D array of scalars
            The plot y-data.
          line_start :: scalar
            The start of the line region in x-data units
          line_end :: scalar
            The end of the line region in x-data units
          amp :: 1-D array of scalars
            The fit Gaussian amplitudes, one element for each
            component
          center :: 1-D array of scalars
            The fit Gaussian centers, one element for each component.
          sigma :: 1-D array of scalars
            The fit Gaussian widths, one element for each component.
          xlabel :: string
            The x-axis label.
          ylabel :: string
            The y-axis label.
          title :: string
            The plot title.
          outfile :: string
            If not None, the filename where to save this figure
          auto :: string
            If False, pause to allow user to review fits.

        Returns: Nothing
        """
        self.ax.clear()
        self.ax.grid(False)
        self.ax.axhline(0,color='k')
        #
        # Plot data
        #
        self.ax.step(xdata,ydata,'k-',where='mid')
        #
        # Plot individual fits
        #
        args = []
        colors = ['b','g','y','c']
        for color,a,c,s in zip(colors,amp,center,sigma):
            if len(amp) == 1:
                color='r'
            if np.any(np.isnan([a,c,s])):
                continue
            args += [a,c,s]
            fwhm = 2.*np.sqrt(2.*np.log(2.))*s
            yfit = gaussian(xdata,a,c,s)
            self.ax.plot(xdata,yfit,color+'-',
                         label='Amp: {0:.2f}; Center: {1:.1f}; FWHM: {2:.1f}'.format(a,c,fwhm))
        #
        # Plot combined fit (if necessary) and residuals
        #
        if not np.any(np.isnan(amp)):
            if len(amp) > 1:
                yfit = gaussian(xdata,*args)
                self.ax.plot(xdata,yfit,'r-',label='Total')
            ind = (xdata > line_start)&(xdata < line_end)
            residuals = ydata-yfit
            self.ax.step(xdata[ind],residuals[ind],'m-',where='mid')
        #
        # Add plot labels
        #
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        xmin = -250
        xmax = 150
        self.ax.set_xlim(xmin,xmax)
        ydata_cut = ydata[np.argmin(np.abs(xdata-xmin)):np.argmin(np.abs(xdata-xmax))]
        yrange = np.max(ydata_cut)-np.min(ydata_cut)
        ymin = np.min(ydata_cut)-0.10*yrange
        ymax = np.max(ydata_cut)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.ax.legend(loc='upper right',fontsize=10)
        self.fig.tight_layout()
        self.fig.savefig(outfile)
        self.fig.show()
        #
        # Wait for user to review
        #
        if not auto:
            print("Click anywhere to continue")
            self.fig.waitforbuttonpress()

def dump_spec(imagename,region,fluxtype):
    """
    Extract spectrum from region.
    
    Inputs: imagename, region, fluxtype
      imagename :: string
        Fits image to analyze
      region :: string
        Region file to use for spectral extraction
      fluxtype :: string
        'total' to measure integrated flux, 'peak' to measure peak
        flux

    Returns: specdata
      specdata :: ndarray
        array with columns 'channel', 'velocity', and 'flux'
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Where the spec data is saved
    #
    logfile = '{0}.{1}.specflux'.format(imagename,region)
    logger.info("Extracting spectrum from {0}".format(imagename))
    logger.info("Dumping spectrum to {0}".format(logfile))
    #
    # Open image, get beam area
    #
    hdu = fits.open(imagename)
    if len(hdu) > 1:
        # need to parse beam table
        image_hdu, beam_hdu = fits.open(imagename)
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
    # Get WCS, channel and velocity axes
    #
    image_wcs = WCS(image_hdu.header)
    wcs_celest = image_wcs.sub(['celestial'])
    channel = np.arange(image_hdu.data.shape[1])
    velocity = ((channel-(image_hdu.header['CRPIX3']-1))*image_hdu.header['CDELT3'] + image_hdu.header['CRVAL3'])/1000. # km/s
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
        pix = wcs_celest.wcs_world2pix(coord.ra.deg,coord.dec.deg,0)
        spec = image_hdu.data[0,:,int(pix[1]),int(pix[0])]*1000. # mJy/beam
    #
    # Read region file, sum spectrum region weighted by continuum
    #
    else:
        region_mask = np.array(fits.open(region)[0].data[0,0],dtype=np.bool)
        cubedata = np.array([chandata*region_mask
                             for chandata in image_hdu.data[0]])
        cubedata = cubedata # Jy/beam
        #
        # Compute weights as median continuum level in each pixel
        #
        weights = np.nanmedian(cubedata,axis=0)
        weights = weights / np.nanmax(weights)
        #
        # Computed weighted sum in each pixel
        #
        spec = np.array([np.nansum(chandata * weights)
                         for chandata in cubedata])
        spec = 1000.* spec / beam_pixel # mJy
        #print(spec)
    #
    # Save spectrum to file
    #
    logfile = '{0}.{1}.specflux'.format(imagename,region)
    with open(logfile,'w') as f:
        f.write('channel velocity flux\n')
        if fluxtype == 'peak':
            f.write('#       km/s     mJy/beam\n')
        else:
            f.write('#       km/s     mJy\n')
        for chan,vel,sp in zip(channel,velocity,spec):
            f.write('{0:7} {1:8.2f} {2:8.4f}\n'.format(chan,vel,sp))
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

def fit_line(title,region,fluxtype,specdata,outfile,auto=False):
    """
    Fit gaussian to RRL.

    Inputs: title, region, fluxtype, specdata, outfile, auto
      title :: string
        title for plot
      region :: string
        region file used to extract spectrum
      fluxtype :: string
        'total' to measure total flux, 'peak' to measure peak flux
      specdata :: ndarray
        output from dump_spec()
      auto :: boolean
        if True, automatically fit spectrum

    Returns:
      line_brightness :: 1-D array of scalars
        line strength (mJy or mJy/beam) for each Gaussian component
      e_line_brightness :: 1-D array of scalars
        line_brightness uncertainty for each component
      line_fwhm :: 1-D array of scalars
        line FWHM (km/s) for each component
      e_line_fwhm :: 1-D array of scalars
        line_fwhm uncertainty for each component
      line_center :: 1-D array of scalars
        line center velocity (km/s) for each component
      e_line_center :: 1-D array of scalars
        line_center uncertainty for each component
      cont_brightness :: scalar
        mean continuum brightness (mJy or mJy/beam) in line-free 
        spectrum
      rms :: scalar
        rms continuum (mJy or mJy/beam) in line-free spectrum
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Make the plot
    #
    myplot = ClickPlot(1)
    if fluxtype=='peak':
        ylabel = 'Flux Density (mJy/beam)'
    else:
        ylabel = 'Flux Density (mJy)'
    #
    # Remove NaNs (if any)
    #
    is_nan = np.isnan(specdata['flux'])
    specdata_flux = specdata['flux'][~is_nan]
    specdata_velocity = specdata['velocity'][~is_nan]
    #
    # Get line-free regions
    #
    title = '{0}\n{1}'.format(title,region)
    if not auto:
        regions = myplot.line_free(specdata_velocity,specdata_flux,
                                   xlabel='Velocity (km/s)',ylabel=ylabel,
                                   title=title)
    else:
        logger.info("Automatically finding line-free regions")
        regions = myplot.auto_line_free(specdata_velocity,specdata_flux,
                                        xlabel='Velocity (km/s)',ylabel=ylabel,
                                        title=title)
        logger.info("Done.")
    #
    # Extract line free velocity and flux
    #
    line_free_mask = np.zeros(specdata_velocity.size,dtype=bool)
    for reg in regions:
        line_free_mask[(specdata_velocity>reg[0])&(specdata_velocity<reg[1])] = True
    line_free_velocity = specdata_velocity[line_free_mask]
    line_free_flux = specdata_flux[line_free_mask]
    #
    # Fit baseline as polyonimal order 3
    #
    logger.info("Fitting continuum baseline polynomial...")
    pfit = np.polyfit(line_free_velocity,line_free_flux,3)
    contfit = np.poly1d(pfit)
    myplot.plot_contfit(specdata_velocity,specdata_flux,contfit,
                        xlabel='Velocity (km/s)',ylabel=ylabel,
                        title=title,auto=auto)
    logger.info("Done.")
    #
    # Subtract continuum
    #
    flux_contsub = specdata_flux - contfit(specdata_velocity)
    line_free_flux_contsub = line_free_flux - contfit(line_free_velocity)
    #
    # Calculate average continuum
    #
    cont_brightness = np.mean(line_free_flux)
    #
    # Compute RMS
    #
    rms = np.sqrt(np.mean(line_free_flux_contsub**2.))
    #
    # Re-plot spectrum, get Gaussian fit estimates, fit Gaussian
    #
    if not auto):
        line_start,center_guesses,sigma_guesses,line_end = \
            myplot.get_gauss(specdata_velocity,flux_contsub,
                             xlabel='Velocity (km/s)',ylabel=ylabel,
                             title=title)
    else:
        logger.info("Automatically estimating Gaussian fit parameters...")
        line_start,center_guesses,sigma_guesses,line_end = \
            myplot.auto_get_gauss(specdata_velocity,flux_contsub,
                                  xlabel='Velocity (km/s)',
                                  ylabel=ylabel,title=title)
        logger.info("Done.")
    #
    # Check that there is a line to fit
    #
    if (None in [line_start,line_end] or None in center_guesses
        or None in sigma_guesses):
        # No line to fit
        line_brightness = np.array([np.nan])
        e_line_brightness = np.array([np.nan])
        line_center = np.array([np.nan])
        e_line_center = np.array([np.nan])
        line_sigma = np.array([np.nan])
        e_line_sigma = np.array([np.nan])
        line_fwhm = np.array([np.nan])
        e_line_fwhm = np.array([np.nan])
    else:
        center_idxs = np.array([np.argmin(np.abs(specdata_velocity-c))
                                for c in center_guesses])
        amp_guesses = flux_contsub[center_idxs]
        #
        # Extract line velocity and fluxes
        #
        line_mask = (specdata_velocity>line_start)&(specdata_velocity<line_end)
        line_flux = flux_contsub[line_mask]
        line_velocity = specdata_velocity[line_mask]
        #
        # Fit gaussian to data
        #
        logger.info("Fitting Gaussian...")
        try:
            p0 = []
            bounds_lower = []
            bounds_upper = []
            for a,c,s in zip(amp_guesses,center_guesses,sigma_guesses):
                p0 += [a,c,s]
                bounds_lower += [0,line_start,0]
                bounds_upper += [np.inf,line_end,np.inf]
            bounds = (bounds_lower,bounds_upper)
            popt,pcov = curve_fit(gaussian,line_velocity,line_flux,
                                  p0=p0,bounds=bounds,
                                  sigma=np.ones(line_flux.size)*rms)
            line_brightness = popt[0::3]
            e_line_brightness = np.sqrt(np.diag(pcov)[0::3])
            line_center = popt[1::3]
            e_line_center = np.sqrt(np.diag(pcov)[1::3])
            line_sigma = np.abs(popt[2::3])
            e_line_sigma = np.sqrt(np.abs(np.diag(pcov)[2::3]))
            line_fwhm = 2.*np.sqrt(2.*np.log(2.))*line_sigma
            e_line_fwhm = 2.*np.sqrt(2.*np.log(2.))*e_line_sigma
        except:
            # Fit failed
            line_brightness = np.array([np.nan])
            e_line_brightness = np.array([np.nan])
            line_center = np.array([np.nan])
            e_line_center = np.array([np.nan])
            line_sigma = np.array([np.nan])
            e_line_sigma = np.array([np.nan])
            line_fwhm = np.array([np.nan])
            e_line_fwhm = np.array([np.nan])
        logger.info("Done.")
    #
    # Plot fit
    #
    myplot.plot_fit(specdata_velocity,flux_contsub,
                    line_start,line_end,
                    line_brightness,line_center,line_sigma,
                    xlabel='Velocity (km/s)',ylabel=ylabel,title=title,
                    outfile=outfile,auto=auto)
    return (line_brightness, e_line_brightness, line_fwhm, e_line_fwhm,
            line_center, e_line_center, cont_brightness, rms)

def calc_rms(ydata):
    """
    Automatically find line-free regions to calculate RMS.

    Inputs: ydata
      ydata :: 1-D array of scalars
        The data

    Returns: rms
      rms :: scalar
        The RMS of line-free regions of ydata
    """
    xdata = np.arange(len(ydata))
    #
    # Get line-free channels by fitting a 3rd order polynomial baseline and
    # rejecting outliers until convergence
    # do this on data smoothed by gaussian 3 channels
    #
    smoy = gaussian_filter(ydata,sigma=5.)
    outliers = np.isnan(ydata)
    while True:
        pfit = np.polyfit(xdata[~outliers],smoy[~outliers],3)
        yfit = np.poly1d(pfit)
        new_smoy = smoy - yfit(xdata)
        rms = 1.4826*np.median(np.abs(new_smoy[~outliers]-np.mean(new_smoy[~outliers])))
        new_outliers = (np.abs(new_smoy) > 3.*rms) | np.isnan(ydata)
        if np.sum(new_outliers) <= np.sum(outliers):
            break
        outliers = new_outliers
    #
    # line-free channels are those without outliers
    #
    rms = np.std(ydata[~outliers])
    return rms

def calc_te(line_brightness, e_line_brightness, line_fwhm, e_line_fwhm,
        line_center, e_line_center, cont_brightness, rms, freq):
    """
    Calculate electron temperature using Balser et al. (2011) equation

    Inputs:
      line_brightness :: scalar
        The RRL intensity (mJy or mJy/beam)
      e_line_brightness :: scalar
        line_brightness uncertainty (mJy or mJy/beam)
      line_fwhm :: scalar
        The RRL FWHM line width (km/s)
      e_line_fwhm :: scalar
        line_fwhm uncertainty (km/s)
      cont_brightness :: scalar
        The continuum brightness (mJy or mJy/beam)
      rms :: scalar
        The continuum RMS (mJy or mJy/beam)
      freq :: scalar
        The RRL frequency (MHz)

    Returns: line_to_cont, e_line_to_cont, te, e_te
      line_to_cont :: scalar
        The line to continuum ratio
      e_line_to_cont :: scalar
        The line_to_cont uncertainty
      te :: scalar
        The electron temperature (K)
      e_te :: scalar
        The electron temperature uncertainty
    """
    #
    # Compute line-to-continuum ratio
    #
    line_to_cont = line_brightness/cont_brightness
    e_line_to_cont = line_to_cont * np.sqrt(rms**2./cont_brightness**2. + e_line_brightness**2./line_brightness**2.)
    #
    # Compute elctron temperature from Balser et al. (2011) using
    # default helium abundance y=0.08
    #
    y = 0.08
    te = (7103.3*(freq/1000.)**1.1/line_to_cont/line_fwhm/(1.+y))**0.87
    e_te = 0.87*te*np.sqrt(e_line_fwhm**2./line_fwhm**2. + rms**2./cont_brightness**2. + e_line_brightness**2./line_brightness**2.)
    return (line_to_cont, e_line_to_cont, te, e_te)

def main(field,regions,spws,pdflabel,stackedspws=[],stackedlabels=[],
         fluxtype='peak',taper=False,imsmooth=False,
         weight=True,outfile='line_info.txt',
         config_file=None,auto=False):
    """
    Extract spectrum from region in each data cube, measure continuum
    brightess and fit Gaussian to measure RRL properties. Also fit stacked 
    lines. Compute electron temperature.

    Inputs:
      field :: string
        The field name
      region :: list of strings
        If only one element, the region file used to analyze each
        spw. Otherwise, the region file to use for each spw.
      spws :: string
        comma-separated string of spws to analyze
      pdflabel :: string
        How to name spectra pdf. Filenames are like
        <pdflabel>.clean.<taper>.pbcor.<imsmooth>.wt.spectra.pdf
      stackedspws :: list of strings
        List of comma-separated spws to stack, one element for each
        stack group
      stackedlabels :: list of strings
        The label for each stack group
      fluxtype :: string
        What type of flux to measure. 'peak' to use peak regions and
        measure peak flux density, 'total' to use full regions and
        measure total flux density.
      taper :: boolean
        if True, use uv-tapered images
      imsmooth :: boolean
        if True, use imsmooth images
      weight :: boolean
        if True, weight the stacked spectra by continuum/rms^2
      outfile :: string
        Filename where the output table is written
      config_file :: string
        The configuration file for this project
      auto :: boolean
        if True, automatically fit a single Gaussian component to
        each spectrum

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
    # If supplied only one region, expand to all spws
    #
    spws = spws.split(',')
    if len(regions) == 1:
        regions = [regions[0] for spw in spws]
    #
    # Get lineid, line frequency for each spw
    #
    alllinespws = config.get("Spectral Windows","Line").split(',')
    alllineids = config.get("Clean","lineids").split(',')
    allrestfreqs = config.get("Clean","restfreqs").split(',')
    #
    # Set-up file
    #
    with open(outfile,'w') as f:
        # 0       1           2      3      4        5        6      7      8        9        10        11          12        13          14
        # lineid  frequency   velo   e_velo line     e_line   fwhm   e_fwhm cont     rms      line2cont e_line2cont elec_temp e_elec_temp linesnr
        # #       MHz         km/s   km/s   mJy/beam mJy/beam km/s   km/s   mJy/beam mJy/beam                       K         K           
        # H122a   9494.152594 -100.0 50.0   1000.00  100.00   100.00 10.00  1000.00  100.00   0.0500    0.0010      10000.0   1000.0      10000.0
        # stacked 9494.152594 -100.0 50.0   1000.00  100.00   100.00 10.00  1000.00  100.00   0.0500    0.0010      10000.0   1000.0      10000.0
        # 1234567 12345678902 123456 123456 12345678 12345678 123456 123456 12345678 12345678 123456789 12345678901 123456789 12345678901 1234567
        #
        headerfmt = '{0:12} {1:12} {2:6} {3:6} {4:8} {5:8} {6:6} {7:6} {8:8} {9:8} {10:9} {11:11} {12:9} {13:11} {14:7}\n'
        rowfmt = '{0:12} {1:12.6f} {2:6.1f} {3:6.1f} {4:8.2f} {5:8.2f} {6:6.2f} {7:6.2f} {8:8.2f} {9:8.2f} {10:9.4f} {11:11.4f} {12:9.1f} {13:11.1f} {14:7.1f}\n'
        f.write(headerfmt.format('lineid','frequency','velo','e_velo',
                                 'line','e_line','fwhm','e_fwhm',
                                 'cont','rms','line2cont','e_line2cont',
                                 'elec_temp','e_elec_temp','linesnr'))
        if fluxtype == 'total':
            fluxunit = 'mJy'
        else:
            fluxunit = 'mJy/beam'
        f.write(headerfmt.format('#','MHz','km/s','km/s',
                                 fluxunit,fluxunit,'km/s','km/s',
                                 fluxunit,fluxunit,'','','K','K',''))
        #
        # Fit RRLs
        #
        goodplots = []
        spws_forstack = []
        specdata_forstack = []
        weights_forstack = []
        for spw,region in zip(spws,regions):
            #
            # Check cube exists
            #
            imagename = '{0}.spw{1}.channel.clean'.format(field,spw)
            if taper: imagename += '.uvtaper'
            imagename += '.pbcor'
            if imsmooth: imagename += '.imsmooth'
            imagename += '.image.fits'
            if not os.path.exists(imagename):
                logger.info("{0} not found.".format(imagename))
                continue
            #
            # Get lineid and restfreq for this spw
            #
            lineid = alllineids[alllinespws.index(spw)]
            restfreq = float(allrestfreqs[alllinespws.index(spw)].replace('MHz',''))
            imagetitle = imagename.replace('spw{0}'.format(spw),lineid)
            outfile = imagename.replace('spw{0}'.format(spw),lineid).replace('.fits','{0}.spec.pdf'.format(region))
            #
            # extract spectrum
            #
            specdata = dump_spec(imagename,region,fluxtype)
            if specdata is None:
                # outside of primary beam
                continue
            #
            # Save data for stacking
            #
            spws_forstack.append(spw)
            specdata_forstack.append(specdata)
            weights_forstack.append(np.nanmean(specdata['flux'])/calc_rms(specdata['flux'])**2.)
            #
            # fit line
            #
            line_brightness, e_line_brightness, line_fwhm, e_line_fwhm, \
              line_center, e_line_center, cont_brightness, rms = \
              fit_line(imagetitle,region,fluxtype,specdata,outfile,auto=auto)
            if line_brightness is None:
                # skipping line
                continue
            #
            # Compute line SNR
            #
            channel_width = config.getfloat("Clean","chanwidth")
            linesnr = 0.7*line_brightness/rms * (line_fwhm/channel_width)**0.5
            #
            # calc Te
            #
            line_to_cont, e_line_to_cont, elec_temp, e_elec_temp = \
              calc_te(line_brightness, e_line_brightness, line_fwhm,
                      e_line_fwhm, line_center, e_line_center,
                      cont_brightness, rms, restfreq)
            #
            # Check crazy, wonky fits if we're in auto mode
            #
            if auto:
                if np.any(line_brightness > 1.e6): # 1000 Jy
                    continue
                if np.any(line_to_cont > 10.):
                    continue
                if np.any(np.isinf(e_line_fwhm)) or np.any(np.isinf(e_line_center)):
                    continue
            #
            # Sort line parameters by brightness
            #
            sortind = np.argsort(line_brightness)[::-1]
            # write line
            multcomps = ['(a)','(b)','(c)','(d)','(e)']
            for multcomp,c,e_c,b,e_b,fw,e_fw,l2c,e_l2c,te,e_te,snr in \
                zip(multcomps,line_center[sortind],e_line_center[sortind],
                    line_brightness[sortind],e_line_brightness[sortind],
                    line_fwhm[sortind],e_line_fwhm[sortind],
                    line_to_cont[sortind],e_line_to_cont[sortind],
                    elec_temp[sortind],e_elec_temp[sortind],
                    linesnr[sortind]):
                if len(line_brightness) == 1:
                    mylineid = lineid
                else:
                    mylineid = lineid+multcomp
                f.write(rowfmt.format(mylineid, restfreq,
                                      c,e_c,b,e_b,fw,e_fw,
                                      cont_brightness, rms,
                                      l2c,e_l2c,te,e_te,snr))
            goodplots.append(outfile)
        #
        # Fit stacked RRLs
        #
        specdata_forstack = np.array(specdata_forstack)
        weights_forstack = np.array(weights_forstack)
        for my_stackedspws, stackedlabel in zip(stackedspws,stackedlabels):
            #
            # Get restfreqs for this stack
            #
            my_stackedspws = my_stackedspws.split(',')
            stack_restfreqs = np.array([float(allrestfreqs[alllinespws.index(spw)].replace('MHz','')) for spw in my_stackedspws
                                        if spw in spws_forstack])
            #
            # Compute average specdata for this stack
            #
            my_stackinds = np.array([spws_forstack.index(spw) for spw in my_stackedspws
                                     if spw in spws_forstack])
            if len(my_stackinds) == 0:
                continue
            if weight:
                specaverage=np.average(specdata_forstack[my_stackinds]['flux'],axis=0,
                                       weights=weights_forstack[my_stackinds])
                stackedrestfreq = np.average(stack_restfreqs,weights=weights_forstack[my_stackinds])
            else:
                specaverage=np.average(specdata_forstack[my_stackinds]['flux'],axis=0)
                stackedrestfreq = np.average(stack_restfreqs)
            #
            # Store stacked spectrum by replacing flux column of
            # one of the original specdata (to retain velocity column)
            #
            avgspecdata = specdata_forstack[my_stackinds][0]
            avgspecdata['flux'] = specaverage
            #
            # Set up output filename and image title
            #
            outfile = '{0}.{1}.channel.clean'.format(field,stackedlabel)
            if taper: outfile += '.uvtaper'
            outfile += '.pbcor'
            if imsmooth: outfile += '.imsmooth'
            outfile += '.{0}'.format(region)
            if weight: outfile += '.wt'
            outfile += '.spec.pdf'
            imagetitle = outfile.replace('.{0}'.format(region),'').replace('.spec.pdf','')
            #
            # fit line
            #
            line_brightness, e_line_brightness, line_fwhm, e_line_fwhm, \
              line_center, e_line_center, cont_brightness, rms = \
              fit_line(imagetitle,region,fluxtype,avgspecdata,outfile,auto=auto)
            if line_brightness is None:
                # skipping line
                continue
            #
            # Compute line SNR
            #
            channel_width = config.getfloat("Clean","chanwidth")
            linesnr = 0.7*line_brightness/rms * (line_fwhm/channel_width)**0.5
            #
            # calc Te
            #
            line_to_cont, e_line_to_cont, elec_temp, e_elec_temp = \
              calc_te(line_brightness, e_line_brightness, line_fwhm,
                      e_line_fwhm, line_center, e_line_center,
                      cont_brightness, rms, stackedrestfreq)
            #
            # Check crazy, wonky fits if we're in auto mode
            #
            if auto:
                if np.any(line_brightness > 1.e6): # 1000 Jy
                    continue
                if np.any(line_to_cont > 10.):
                    continue
                if np.any(np.isinf(e_line_fwhm)) or np.any(np.isinf(e_line_center)):
                    continue
            #
            # Sort line parameters by line_brightness
            #
            sortind = np.argsort(line_brightness)[::-1]
            # write line
            multcomps = ['(a)','(b)','(c)','(d)','(e)']
            for multcomp,c,e_c,b,e_b,fw,e_fw,l2c,e_l2c,te,e_te,snr in \
                zip(multcomps,line_center[sortind],e_line_center[sortind],
                    line_brightness[sortind],e_line_brightness[sortind],
                    line_fwhm[sortind],e_line_fwhm[sortind],
                    line_to_cont[sortind],e_line_to_cont[sortind],
                    elec_temp[sortind],e_elec_temp[sortind],
                    linesnr[sortind]):
                if len(line_brightness) == 1:
                    mylineid = stackedlabel
                else:
                    mylineid = stackedlabel+multcomp
                f.write(rowfmt.format(mylineid, stackedrestfreq,
                                      c,e_c,b,e_b,fw,e_fw,
                                      cont_brightness, rms,
                                      l2c,e_l2c,te,e_te,snr))
            goodplots.append(outfile)
    #
    # Generate TeX file of all plots
    #
    logger.info("Generating PDF...")
    # fix filenames so LaTeX doesn't complain
    plots = ['{'+fn.replace('.pdf','')+'}.pdf' for fn in goodplots]
    fname = '{0}.clean'.format(pdflabel)
    if taper: fname += '.uvtaper'
    fname += '.pbcor'
    if imsmooth: fname += '.imsmooth'
    if weight: fname += '.wt'
    fname += '.spectra.tex'
    with open(fname,'w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        for i in range(0,len(plots),6):
            f.write(r"\begin{figure}"+"\n")
            f.write(r"\centering"+"\n")
            if len(plots) > i: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i]+"}\n")
            if len(plots) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+3]+"}\n")
            if len(plots) > i+1: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+1]+"}\n")
            if len(plots) > i+4: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+4]+"}\n")
            if len(plots) > i+2: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+2]+"}\n")
            if len(plots) > i+5: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+5]+"}\n")
            f.write(r"\end{figure}"+"\n")
            f.write(r"\clearpage"+"\n")
        f.write(r"\end{document}")
    os.system('pdflatex -interaction=batchmode {0}'.format(fname))
    logger.info("Done.")
