""" Scripts for generating galaxy models """

from scipy.stats import gaussian_kde
from scipy.special import gamma, gammainc, gammaincinv, kn
from scipy.optimize import newton

from numpy import pi, exp, mgrid, array, sin, cos, cosh, sqrt
from numpy.random import uniform

from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.models import Sersic2D

import galprime

def object_kde(columns):
    return gaussian_kde(columns)


def i_at_r50(mag, n=2, r_50=2, m_0=27):
    """ Get the intensity at the half-light radius """
    b_n = b(n)
    l_tot = 10 ** ((mag - m_0) / -2.5) * (b_n ** (2 * n))
    denom = (r_50 ** 2) * 2 * pi * n * exp(b_n) * gamma(2 * n)
    i_e = l_tot / denom

    return i_e

def b(n, estimate=False):
    """ Get the b_n normalization constant for the sersic profile. 
    From Graham and Driver.
    """
    if estimate:
        return 2 * n - (1 / 3) + (4 / (405 * n)) + (46 / (25515 * (n ** 2)))
    else:
        return gammaincinv(2 * n, 0.5)


def model_from_kde(kde, config=None, mag_kde=None, names=None, seed=None):
    
    cutout_size = 101 if config is None else config["SIZE"]
    arc_conv = 1 if config is None else config["ARC_CONV"]
    
    attempt = array(kde.resample(1, seed=seed), dtype=float)
    params = galprime.ordered_dict(names, attempt)
    
    if mag_kde is not None:
        params["MAGS"] = mag_kde.resample(1, seed=seed)[0]
    
    r_50_pix = params["R50S"] / arc_conv
    
    i_R50 = i_at_r50(params["MAGS"], n=params["NS"], r_50=r_50_pix, m_0=config["ZEROPOINT"])
    theta = uniform(0, 2*pi)
    
    params["I_R50"] = i_R50
    params["PA"] = theta
    params["R50_PIX"] = params["R50S"] / arc_conv
    
    x,y = mgrid[:cutout_size, :cutout_size]
    # Generate the Sersic model from the parameters
    sersic_model = gen_sersic_model(i_R50=i_R50, r_eff=params["R50S"], n=params["NS"], ellip=params["ELLIPS"], 
                                    theta=theta, x_0=cutout_size / 2, y_0=cutout_size/2, 
                                    arc_conv=arc_conv, shape=(cutout_size, cutout_size))
    
    for n in params:
        params[n] = float(params[n])


    return sersic_model, params
    

def gen_sersic_model(i_R50=10, r_eff=5, n=1, ellip=0.1, theta=0, x_0=0, y_0=0, arc_conv=1, shape=(51,51)):
    try:
        sersic_model = Sersic2D(amplitude=i_R50, r_eff=r_eff / arc_conv, n=n, ellip=ellip, theta=theta, x_0=x_0, y_0=y_0)
        x,y = mgrid[:shape[0], :shape[1]]
        z = sersic_model(x,y)

    except ValueError:
        attempt = [i_R50, r_eff, n, ellip, theta, x_0, y_0, arc_conv, shape]
        raise galprime.GalPrimeError("Failed to generate sersic model with the following parameters:\n" + str(attempt))
    
    return z


def check_sersic_params(i_r50, i_r50_max=200):
    pass


def core_sersic_b(n, r_b, r_e):
    """
    Get the scale parameter b using the Newton-Raphson root finder.
    :param n: Sersic index
    :param r_b: Break radius
    :param r_e: Effective radius
    :return:
    """
    # Start by getting an initial guess at b (using the regular Sersic estimation)
    b_guess = 2 * n - (1 / 3)

    # Define the combination of gamma functions that makes up the relation
    # We want to find the zeroes of this.
    def evaluate(b_in):
        comp1 = gamma(2 * n)
        comp2 = gammainc(2 * n, b_in * ((r_b / r_e) ** (1 / n)))
        comp3 = 2 * gammainc(2 * n, b_in)

        return comp1 + comp2 - comp3

    return newton(evaluate, x0=b_guess)


class EdgeOnDisk(Fittable2DModel):
    """
    Two-dimensional Edge-On Disk model.

    Parameters
    ----------
    amplitude : float
        Brightness at galaxy centre
    scale_x : float
        Scale length along the semi-major axis
    scale_y : float
        Scale length along the semi-minor axis
    x_0 : float, optional
        x position of the center
    y_0 : float, optional
        y position of the center
    theta: float, optional
        Position angle in radians, counterclockwise from the
        positive x-axis.
    """
    amplitude = Parameter(default=1)
    scale_x = Parameter(default=1)
    scale_y = Parameter(default=0.5)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    theta = Parameter(default=0)

    @classmethod
    def evaluate(cls, x, y, amplitude, scale_x, scale_y, x_0, y_0, theta):
        """Exaluate model on a 2D x-y grid."""

        x_maj = abs((x - x_0) * cos(theta) + (y - y_0) * sin(theta))
        x_min = -(x - x_0) * sin(theta) + (y - y_0) * cos(theta)

        return amplitude * (x_maj / scale_x) * kn(1, x_maj / scale_x) / (cosh(x_min / scale_y) ** 2)


class Core_Sersic(Fittable2DModel):
    """
    Two-dimensional Edge-On Disk model.

    Parameters
    ----------
    r_e : float
        Effective radius of the galaxy.
    r_b : float
        Break Radius (Where the model switches from one regime to the other).
    I_b : float
        Intensity at the break radius
    alpha : float
        Defines the "sharpness" of the model transitions
    gamma : float
        Power law slope
    n     : float
        Sersic index (see info on Sersic profiles if needed)
    x_0 : float, optional
        x position of the center
    y_0 : float, optional
        y position of the center
    theta: float, optional
        Position angle in radians, counterclockwise from the positive x-axis.
    ellip: float, optional
        Ellipticity of the model (default is 0 : circular)


    """

    r_e = Parameter(default=5)
    r_b = Parameter(default=1)
    I_b = Parameter(default=1)
    n = Parameter(default=1)
    alpha = Parameter(default=1)
    gamma = Parameter(default=1)

    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    theta = Parameter(default=0)
    ellip = Parameter(default=0)

    @classmethod
    def evaluate(cls, x, y, r_e, r_b, I_b, n, alpha, gamma, x_0, y_0, theta, ellip):
        """Two dimensional Core-Sersic profile function."""

        bn = core_sersic_b(n, r_b, r_e)

        def core_sersic_i_prime():
            return I_b * (2 ** (- gamma / alpha)) * exp(bn * (2 ** (1 / (alpha * n))) * (r_b / r_e) ** (1 / n))

        i_prime = core_sersic_i_prime()

        cos_theta, sin_theta = cos(theta), sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = sqrt(x_maj ** 2 + (x_min / (1-ellip)) ** 2)

        comp_1 = i_prime * ((1 + ((r_b / z) ** alpha)) ** (gamma / alpha))
        comp_2 = -bn * (((z ** alpha) + (r_b ** alpha)) / (r_e ** alpha)) ** (1 / (n * alpha))

        return comp_1 * exp(comp_2)
