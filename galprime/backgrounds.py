""" Background estimation and subtraction module.

This module contains all methods available to estimate the backgrounds of cutouts,
and to subtract these backgrounds from cutouts or profiles as a means to correct them.

"""

import numpy as np
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.convolution import convolve, Tophat2DKernel, Gaussian2DKernel

from photutils import Background2D, MedianBackground, BkgZoomInterpolator
from photutils.segmentation import detect_threshold, detect_sources


def make_source_mask(data, nsigma, npixels, mask=None, filter_fwhm=None,
                     filter_size=3, kernel=None, sigclip_sigma=3.0,
                     sigclip_iters=5, dilate_size=11):
    """
        Source mask generation (from photutils 1.4) 
    """ 
    from scipy import ndimage

    threshold = detect_threshold(data, nsigma, background=None, error=None,
                                 mask=mask, sigclip_sigma=sigclip_sigma,
                                 sigclip_iters=sigclip_iters)

    if kernel is None and filter_fwhm is not None:
        kernel_sigma = filter_fwhm * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(kernel_sigma, x_size=filter_size,
                                  y_size=filter_size)
    if kernel is not None:
        kernel.normalize()

    segm = detect_sources(data, threshold, npixels, kernel=kernel)
    if segm is None:
        return np.zeros(data.shape, dtype=bool)

    selem = np.ones((dilate_size, dilate_size))
    return ndimage.binary_dilation(segm.data.astype(bool), selem)



def background_2D(img, mask, box_size, interp=None, filter_size=1,
                  exclude_percentile=90):
    """ Run photutils background with SigmaClip and MedianBackground

    :param img: The 2D input image
    :type img: array_like (np.ndarray)
    :param mask: The 2D image mask
    :type mask: array_like (np.ndarray)
    :param box_size: The size of the box used in the 2D background. It should be larger than
        the largest objects in your image but still sufficiently small to capture large-scale
        structure.
    :type box_size: int, optional
    :param filter_size: The window size of the median filter being applied to the background image. A
        higher filter size will result in more smoothing to the background.
    :type filter_size: int, optional
    :param exclude_percentile: If the percentage of masked pixels in a box is above the exclude percentile,
        it is not included in determining the 2D background.
    :type exclude_percentile: float, optional
    """
    if interp is None:
        interp = BkgZoomInterpolator()
    return Background2D(img, box_size,
                        sigma_clip=SigmaClip(sigma=3.),
                        filter_size=filter_size,
                        bkg_estimator=MedianBackground(),
                        exclude_percentile=exclude_percentile,
                        mask=mask,
                        interpolator=interp)


class SourceMask:
    def __init__(self, img, nsigma=3., npixels=10, mask=None):
        """ Helper for making & dilating a source mask.
             See Photutils docs for make_source_mask.

            :param img: The image that is to be masked
            :type img: array_like (np.ndarray)
            :param nsigma: The sigma detection threshold for the source mask, defaults to 3
            :type nsigma: float, optional
            :param npixels: The number of required pixels for a detection, defaults to 10
            :type npixels: int, optional
            :param mask: An already-suppled mask for objects in the image.
            :type array_like (np.ndarray), optional
        """
        self.img = img
        self.nsigma = nsigma
        self.npixels = npixels
        if mask is None:
            self.mask = np.zeros(self.img.shape, dtype=np.bool)
        else:
            self.mask = mask

    def single(self, filter_fwhm=3., tophat_size=5., mask=None):
        """ Mask on a single scale """
        if mask is None:
            image = self.img
        else:
            image = self.img * (1 - mask)
        mask = make_source_mask(image, nsigma=self.nsigma,
                                npixels=self.npixels,
                                dilate_size=1, filter_fwhm=filter_fwhm)
        return dilate_mask(mask, tophat_size)

    def multiple(self, filter_fwhm=[3.], tophat_size=[3.], mask=None):
        """ Mask repeatedly on different scales """
        if mask is None:
            self.mask = np.zeros(self.img.shape, dtype=np.bool)
        for fwhm, tophat in zip(filter_fwhm, tophat_size):
            smask = self.single(filter_fwhm=fwhm, tophat_size=tophat)
            self.mask = self.mask | smask  # Or the masks at each iteration

        return self.mask

    def dilated(self, tophat_size=[3.], mask=None):
        """Mask using simple dilation"""
        if mask is None:
            self.mask = self.single()
        for tophat in tophat_size:
            smask = dilate_mask(self.mask, tophat)
            self.mask = self.mask | smask  # Or the masks at each iteration

        return self.mask


def dilate_mask(mask, tophat_size):
    """ Dilate a mask with a tophat kernel. """
    area = np.pi * tophat_size ** 2.
    kernel = Tophat2DKernel(tophat_size)
    dilated_mask = convolve(mask, kernel) >= 1. / area
    return dilated_mask


def estimate_background_sigclip(cutout, config=None, nsigma=2, npixels=3, dilate_size=7):
    """ Estimate the background mean, median, and standard deviation of a cutout using sigma-clipped-stats """

    nsigma is config["MASK_PARAMS"][0] if config is not None else nsigma
    npixels is config["MASK_PARAMS"][2] if config is not None else npixels

    bg_mask = make_source_mask(cutout, nsigma=nsigma, npixels=npixels, dilate_size=dilate_size)

    bg_mean, bg_median, bg_std = sigma_clipped_stats(cutout, sigma=3.0, mask=bg_mask)

    return bg_mean, bg_median, bg_std

