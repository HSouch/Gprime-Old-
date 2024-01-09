from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
from photutils import detect_threshold, detect_sources, deblend_sources


class MaskedCutout:
    def __init__(self, cutout=None, mask=None, masked_cutout=None, config=None, 
                 segment_array=None, mask_info = None):
        self.cutout = cutout
        self.mask = mask
        self.segment_array = segment_array
        self.masked_cutout = masked_cutout
        self.mask_info = {} if mask_info is None else mask_info
        
        if config is not None:
            params = config["MASK_PARAMS"]
        else:
            params = [1., 2.0, 11]
        
        self.nsigma, self.gauss_width, self.npixels = params
        
    def gen_mask(self, nsigma=None, gauss_width=None, npixels=None, omit_centre=True, deblend=True, omit=None):
        nsigma = self.nsigma if nsigma is None else nsigma
        gauss_width = self.gauss_width if gauss_width is None else gauss_width
        npixels = self.npixels if npixels is None else npixels
        
        sigma = gauss_width * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma)
        
        # First get our segment array (deblended or not)
        threshold = detect_threshold(self.cutout, nsigma=nsigma)
        segments = detect_sources(self.cutout, threshold, npixels=npixels, kernel=kernel)
        
        if deblend:
            try:
                segments = deblend_sources(self.cutout, segments, npixels=npixels, kernel=kernel, progress_bar=False)
            except ImportError:
                print("Skimage not working!")
        
        self.segment_array = segments
        
        # Now we apply the mask
        if omit is None:
            if omit_centre:
                segments_shape = self.segment_array.data.shape
                centre_value = self.segment_array.data[int(segments_shape[0]/2), int(segments_shape[1]/2)]
                omit = [0, centre_value]
            else:
                omit=[0]
        
        self.mask = np.isin(self.segment_array.data, omit, invert=True).astype(int)
        n_masked = np.sum(self.mask)

        self.mask_info["NSEG"] = self.segment_array.nlabels
        self.mask_info["CENT"] = centre_value
        self.mask_info["N_MASKED"] = n_masked
        self.mask_info["P_MASKED"] = n_masked / (self.cutout.shape[0] * self.cutout.shape[1])
        
    def apply_mask(self):
        self.masked_cutout = np.copy(self.cutout)
        self.masked_cutout[self.mask == 1] = np.nan
        return self.masked_cutout
    
    def mask_cutout(self, nsigma=None, gauss_width=None, npixels=None, omit_centre=True, deblend=True, omit=None):
        # If a mask has not been generated yet, generate the mask
        if self.mask is None:
            self.gen_mask(nsigma=nsigma, gauss_width=gauss_width, npixels=npixels, 
                     omit_centre=omit_centre, deblend=deblend, omit=omit)
        
        self.apply_mask()
