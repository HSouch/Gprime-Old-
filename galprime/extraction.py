import sys
from galprime.isophote_l import Ellipse, EllipseGeometry
from numpy import max, pi, log, unravel_index, argmax, ceil
from photutils.morphology import data_properties


def isophote_fitting(data, config=None, centre_method='standard'):
    """ Wrapper for the photutils isophote extraction routine. Designed to get the most out of profile 
        extractions.

    :param data: Input data (2D cutout)
    :type data: numpy.ndarray
    :param config: The configuration file with isophote fitting params. Will use default parameters if 
        nothing is supplied, defaults to None
    :type config: dict, optional
    :param centre_method: Which method to use when finding the centre for initial isophote, defaults to 'standard'
        'standard'  Uses the central region of the cutout
        'max'       Uses the location of the brightest pixel. Only use if your cutout is properly masked.
    :type centre_method: str, optional
    :return: The photutils isophotelist object containing extraction results.
    :rtype: photutils.isophote.IsophoteList
    """

    fail_count, max_fails = 0, 1000
    linear = False if config is None else config["LINEAR"]
    step = 1. if config is None else config["STEP"]

    # Get centre of image and cutout halfwidth
    if centre_method == 'standard':
        centre = (data.shape[0]/2, data.shape[1]/2)
    elif centre_method == 'max':
        centre = unravel_index(argmax(data), data.shape)
    else:
        centre = (data.shape[0] / 2, data.shape[1] / 2)

    # Here we can just extract out to the requested size if specified in the cutout
    cutout_halfwidth = max((ceil(data.shape[0] / 2), ceil(data.shape[1] / 2)))
    if config is not None:
        cutout_halfwidth = config["EXTRACTION_SIZE"] / 2
    
    fitting_list = []

    # First, try obtaining morphological properties from the data and fit using that starting ellipse
    try:
        morph_cat = data_properties(log(data))
        r = 2.0
        pos = (morph_cat.xcentroid, morph_cat.ycentroid)
        a = morph_cat.semimajor_sigma.value * r
        b = morph_cat.semiminor_sigma.value * r
        theta = morph_cat.orientation.value

        geometry = EllipseGeometry(pos[0], pos[1], sma=a, eps=(1 - (b / a)), pa=theta)
        flux = Ellipse(data, geometry)
        fitting_list = flux.fit_image(maxit=100, maxsma=cutout_halfwidth, step=step, linear=linear,
                                      maxrit=cutout_halfwidth / 2)
        if len(fitting_list) > 0:
            return fitting_list

    except KeyboardInterrupt:
        sys.exit(1)
    except (RuntimeError, ValueError, OverflowError, IndexError):
        fail_count += 1
        if fail_count >= max_fails:
            return []

    # If that fails, test a parameter space of starting ellipses
    try:
        for angle in range(0, 180, 45):
            for sma in range(2, 26, 5):
                for eps in (0.3, 0.5, 0.9):
                    geometry = EllipseGeometry(float(centre[0]), float(centre[1]), eps=eps,
                                               sma=sma, pa=angle * pi / 180.)
                    flux = Ellipse(data, geometry)
                    fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=step, linear=linear,
                                                  maxrit=cutout_halfwidth / 3)
                    if len(fitting_list) > 0:
                        return fitting_list

    except KeyboardInterrupt:
        sys.exit(1)
    except (RuntimeError, ValueError, OverflowError, IndexError):

        # print("RuntimeError or ValueError")
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except IndexError:
        fail_count += 1
        if fail_count >= max_fails:
            return []

    return fitting_list
