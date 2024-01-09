# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides a class to sample data along an elliptical path.
"""

import copy

import numpy as np

from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import INTEGRATORS

__all__ = ['EllipseSample']


class EllipseSample:

    def __init__(self, image, sma, x0=None, y0=None, astep=0.1, eps=0.2,
                 position_angle=0.0, sclip=3.0, nclip=0, linear_growth=False,
                 integrmode='bilinear', geometry=None):
        self.image = image
        self.integrmode = integrmode

        if geometry:
            # when the geometry is inherited from somewhere else,
            # its sma attribute must be replaced by the value
            # explicitly passed to the constructor.
            self.geometry = copy.deepcopy(geometry)
            self.geometry.sma = sma
        else:
            # if no center was specified, assume it's roughly
            # coincident with the image center
            _x0 = x0
            _y0 = y0
            if not _x0 or not _y0:
                _x0 = image.shape[1] / 2
                _y0 = image.shape[0] / 2

            self.geometry = EllipseGeometry(_x0, _y0, sma, eps,
                                            position_angle, astep,
                                            linear_growth)

        # sigma-clip parameters
        self.sclip = sclip
        self.nclip = nclip

        # extracted values associated with this sample.
        self.values = None
        self.mean = None
        self.gradient = None
        self.gradient_error = None
        self.gradient_relative_error = None
        self.sector_area = None

        # total_points reports the total number of pairs angle-radius that
        # were attempted. actual_points reports the actual number of sampled
        # pairs angle-radius that resulted in valid values.
        self.total_points = 0
        self.actual_points = 0

    def extract(self):
        """
        Extract sample data by scanning an elliptical path over the
        image array.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The rows of the array contain the angles, radii, and
            extracted intensity values, respectively.
        """
        # the sample values themselves are kept cached to prevent
        # multiple calls to the integrator code.
        if self.values is not None:
            return self.values
        else:
            s = self._extract()
            self.values = s
            return s

    def _extract(self, phi_min=0.05):

        angles = []
        radii = []
        intensities = []
        sector_areas = []

        # reset counters
        self.total_points = 0
        self.actual_points = 0

        # build integrator
        integrator = INTEGRATORS[self.integrmode](self.image, self.geometry,
                                                  angles, radii, intensities)

        # initialize walk along elliptical path
        radius = self.geometry.initial_polar_radius
        phi = self.geometry.initial_polar_angle


        if integrator.is_area():
            integrator.integrate(radius, phi)
            area = integrator.get_sector_area()
            # this integration that just took place messes up with the
            # storage arrays and the constructors. We have to build a new
            # integrator instance from scratch, even if it is the same
            # kind as originally selected by the caller.
            angles = []
            radii = []
            intensities = []
            if area < 1.0:
                integrator = INTEGRATORS['bilinear'](
                    self.image, self.geometry, angles, radii, intensities)
            else:
                integrator = INTEGRATORS[self.integrmode](self.image,
                                                          self.geometry,
                                                          angles, radii,
                                                          intensities)

        # walk along elliptical path, integrating at specified
        # places defined by polar vector. Need to go a bit beyond
        # full circle to ensure full coverage.
        while phi <= np.pi * 2.0 + phi_min:
            # do the integration at phi-radius position, and append
            # results to the angles, radii, and intensities lists.
            integrator.integrate(radius, phi)

            # store sector area locally
            sector_areas.append(integrator.get_sector_area())

            # update total number of points
            self.total_points += 1

            # update angle and radius to be used to define
            # next polar vector along the elliptical path
            phistep_ = integrator.get_polar_angle_step()
            phi += min(phistep_, 0.5)
            radius = self.geometry.radius(phi)

        # average sector area is calculated after the integrator had
        # the opportunity to step over the entire elliptical path.
        self.sector_area = np.mean(np.array(sector_areas))

        # apply sigma-clipping.
        angles, radii, intensities = self._sigma_clip(angles, radii,
                                                      intensities)

        # actual number of sampled points, after sigma-clip removed outliers.
        self.actual_points = len(angles)

        # pack results in 2-d array
        result = np.array([np.array(angles), np.array(radii),
                           np.array(intensities)])

        return result

    def _sigma_clip(self, angles, radii, intensities):
        if self.nclip > 0:
            for _ in range(self.nclip):
                # do not use list.copy()! must be python2-compliant.
                angles, radii, intensities = self._iter_sigma_clip(
                    angles[:], radii[:], intensities[:])

        return np.array(angles), np.array(radii), np.array(intensities)

    def _iter_sigma_clip(self, angles, radii, intensities):
        # Can't use scipy or astropy tools because they use masked arrays.
        # Also, they operate on a single array, and we need to operate on
        # three arrays simultaneously. We need something that physically
        # removes the clipped points from the arrays, since that is what
        # the remaining of the `ellipse` code expects.
        r_angles = []
        r_radii = []
        r_intensities = []

        values = np.array(intensities)
        mean = np.mean(values)
        sig = np.std(values)
        lower = mean - self.sclip * sig
        upper = mean + self.sclip * sig

        count = 0
        for k in range(len(intensities)):
            if intensities[k] >= lower and intensities[k] < upper:
                r_angles.append(angles[k])
                r_radii.append(radii[k])
                r_intensities.append(intensities[k])
                count += 1

        return r_angles, r_radii, r_intensities

    def update(self, fixed_parameters=None):
        """
        Update this `~photutils.isophote.EllipseSample` instance.

        This method calls the
        :meth:`~photutils.isophote.EllipseSample.extract` method to get
        the values that match the current ``geometry`` attribute, and
        then computes the the mean intensity, local gradient, and other
        associated quantities.
        """
        if fixed_parameters is None:
            fixed_parameters = np.array([False, False, False, False])
        self.geometry.fix = fixed_parameters

        step = self.geometry.astep

        # Update the mean value first, using extraction from main sample.
        s = self.extract()
        self.mean = np.mean(s[2])

        # Get sample with same geometry but at a different distance from
        # center. Estimate gradient from there.
        gradient, gradient_error = self._get_gradient(step)

        # Check for meaningful gradient. If no meaningful gradient, try
        # another sample, this time using larger radius. Meaningful
        # gradient means something  shallower, but still close to within
        # a factor 3 from previous gradient estimate. If no previous
        # estimate is available, guess it by adding the error to the
        # current gradient.
        previous_gradient = self.gradient
        if not previous_gradient:
            previous_gradient = gradient + gradient_error

            # solution adopted before 08/12/2019
            # previous_gradient = -0.05  # good enough, based on usage

        if gradient >= (previous_gradient / 3.0):  # gradient is negative!
            gradient, gradient_error = self._get_gradient(2 * step)

        # If still no meaningful gradient can be measured, try with
        # previous one, slightly shallower. A factor 0.8 is not too far
        # from what is expected from geometrical sampling steps of 10-20%
        # and a deVaucouleurs law or an exponential disk (at least at its
        # inner parts, r <~ 5 req). Gradient error is meaningless in this
        # case.
        if gradient >= (previous_gradient / 3.0):
            gradient = previous_gradient * 0.8
            gradient_error = None

        self.gradient = gradient
        self.gradient_error = gradient_error
        if gradient_error and gradient < 0.0:
            self.gradient_relative_error = gradient_error / np.abs(gradient)
        else:
            self.gradient_relative_error = None

    def _get_gradient(self, step):
        gradient_sma = (1.0 + step) * self.geometry.sma

        gradient_sample = EllipseSample(
            self.image, gradient_sma, x0=self.geometry.x0,
            y0=self.geometry.y0, astep=self.geometry.astep, sclip=self.sclip,
            nclip=self.nclip, eps=self.geometry.eps,
            position_angle=self.geometry.pa,
            linear_growth=self.geometry.linear_growth,
            integrmode=self.integrmode)

        sg = gradient_sample.extract()
        mean_g = np.mean(sg[2])
        gradient = (mean_g - self.mean) / self.geometry.sma / step

        s = self.extract()
        sigma = np.std(s[2])
        sigma_g = np.std(sg[2])

        gradient_error = (np.sqrt(sigma**2 / len(s[2])
                                  + sigma_g**2 / len(sg[2]))
                          / self.geometry.sma / step)

        return gradient, gradient_error

    def coordinates(self):
        """
        Return the (x, y) coordinates associated with each sampled
        point.

        Returns
        -------
        x, y : 1D `~numpy.ndarray`
            The x and y coordinate arrays.
        """
        angles = self.values[0]
        radii = self.values[1]
        x = np.zeros(len(angles))
        y = np.zeros(len(angles))

        for i in range(len(x)):
            x[i] = (radii[i] * np.cos(angles[i] + self.geometry.pa)
                    + self.geometry.x0)

            y[i] = (radii[i] * np.sin(angles[i] + self.geometry.pa)
                    + self.geometry.y0)

        return x, y


class CentralEllipseSample(EllipseSample):
    """
    An `~photutils.isophote.EllipseSample` subclass designed to handle
    the special case of the central pixel in the galaxy image.
    """

    def update(self, fixed_parameters):
        """
        Update this `~photutils.isophote.EllipseSample` instance with
        the intensity integrated at the (x0, y0) center position using
        bilinear integration. The local gradient is set to `None`.

        'fixed_parameters' is ignored in this subclass.
        """
        s = self.extract()
        self.mean = s[2][0]

        self.gradient = None
        self.gradient_error = None
        self.gradient_relative_error = None

    def _extract(self):
        angles = []
        radii = []
        intensities = []

        integrator = INTEGRATORS['bilinear'](self.image, self.geometry,
                                             angles, radii, intensities)
        integrator.integrate(0.0, 0.0)

        self.total_points = 1
        self.actual_points = 1

        return np.array([np.array(angles), np.array(radii),
                         np.array(intensities)])
