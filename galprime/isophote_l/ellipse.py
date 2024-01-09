# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides a class to fit elliptical isophotes.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.isophote.fitter import (DEFAULT_CONVERGENCE, DEFAULT_FFLAG,
                                       DEFAULT_MAXGERR, DEFAULT_MAXIT,
                                       DEFAULT_MINIT, CentralEllipseFitter,
                                       EllipseFitter)
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import BILINEAR
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.sample import CentralEllipseSample, EllipseSample

__all__ = ['Ellipse']


class Ellipse:

    def __init__(self, image, geometry=None, threshold=0.1):
        self.image = image

        if geometry is not None:
            self._geometry = geometry
        else:
            _x0 = image.shape[1] / 2
            _y0 = image.shape[0] / 2
            self._geometry = EllipseGeometry(_x0, _y0, 10.0, eps=0.2,
                                             pa=np.pi / 2)
        self.set_threshold(threshold)

    def set_threshold(self, threshold):
        """
        Modify the threshold value used by the centerer.

        Parameters
        ----------
        threshold : float
            The new threshold value to use.
        """
        self._geometry.centerer_threshold = threshold

    def fit_image(self, sma0=None, minsma=0.0, maxsma=None, step=0.1,
                  conver=DEFAULT_CONVERGENCE, minit=DEFAULT_MINIT,
                  maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG,
                  maxgerr=DEFAULT_MAXGERR, sclip=3.0, nclip=0,
                  integrmode=BILINEAR, linear=None, maxrit=None,
                  fix_center=False, fix_pa=False, fix_eps=False):
        # This parameter list is quite large and should in principle be
        # simplified by re-distributing these controls to somewhere else.
        # We keep this design though because it better mimics the flat
        # architecture used in the original STSDAS task `ellipse`.
        
        # multiple fitted isophotes will be stored here
        isophote_list = []

        # get starting sma from appropriate source: keyword parameter,
        # internal EllipseGeometry instance, or fixed default value.
        if not sma0:
            if self._geometry:
                sma = self._geometry.sma
            else:
                sma = 10.0
        else:
            sma = sma0

        # Override geometry instance with parameters set at the call.
        if isinstance(linear, bool):
            self._geometry.linear_growth = linear
        else:
            linear = self._geometry.linear_growth
        if fix_center and fix_pa and fix_eps:
            warnings.warn(': Everything is fixed. Fit not possible.',
                          AstropyUserWarning)
            return IsophoteList([])
        if fix_center or fix_pa or fix_eps:
            # Note that this overrides the geometry instance for good.
            self._geometry.fix = np.array([fix_center, fix_center, fix_pa,
                                           fix_eps])

        # first, go from initial sma outwards until
        # hitting one of several stopping criteria.
        noiter = False
        first_isophote = True
        while True:
            # first isophote runs longer
            minit_a = 2 * minit if first_isophote else minit
            first_isophote = False

            isophote = self.fit_isophote(sma, step, conver, minit_a, maxit,
                                         fflag, maxgerr, sclip, nclip,
                                         integrmode, linear, maxrit,
                                         noniterate=noiter,
                                         isophote_list=isophote_list)

            # check for failed fit.
            if isophote.stop_code < 0 or isophote.stop_code == 1:
                # in case the fit failed right at the outset, return an
                # empty list. This is the usual case when the user
                # provides initial guesses that are too way off to enable
                # the fitting algorithm to find any meaningful solution.

                if len(isophote_list) == 1:
                    warnings.warn('No meaningful fit was possible.',
                                  AstropyUserWarning)
                    return IsophoteList([])

                self._fix_last_isophote(isophote_list, -1)

                # get last isophote from the actual list, since the last
                # `isophote` instance in this context may no longer be OK.
                isophote = isophote_list[-1]

                # if two consecutive isophotes failed to fit,
                # shut off iterative mode. Or, bail out and
                # change to go inwards.
                if len(isophote_list) > 2:
                    if ((isophote.stop_code == 5
                         and isophote_list[-2].stop_code == 5)
                            or isophote.stop_code == 1):
                        if maxsma and maxsma > isophote.sma:
                            # if a maximum sma value was provided by
                            # user, and the current sma is smaller than
                            # maxsma, keep growing sma in non-iterative
                            # mode until reaching it.
                            noiter = True
                        else:
                            # if no maximum sma, stop growing and change
                            # to go inwards.
                            break

            # reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]

            # update sma. If exceeded user-defined
            # maximum, bail out from this loop.
            sma = isophote.sample.geometry.update_sma(step)
            if maxsma and sma >= maxsma:
                break

        # reset sma so as to go inwards.
        first_isophote = isophote_list[0]
        sma, step = first_isophote.sample.geometry.reset_sma(step)

        # now, go from initial sma inwards towards center.
        while True:
            isophote = self.fit_isophote(sma, step, conver, minit, maxit,
                                         fflag, maxgerr, sclip, nclip,
                                         integrmode, linear, maxrit,
                                         going_inwards=True,
                                         isophote_list=isophote_list)

            # if abnormal condition, fix isophote but keep going.
            if isophote.stop_code < 0:
                self._fix_last_isophote(isophote_list, 0)

            # but if we get an error from the scipy fitter, bail out
            # immediately. This usually happens at very small radii
            # when the number of data points is too small.
            if isophote.stop_code == 3:
                break

            # reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]

            # figure out next sma; if exceeded user-defined
            # minimum, or too small, bail out from this loop
            sma = isophote.sample.geometry.update_sma(step)
            if sma <= max(minsma, 0.5):
                break

        # if user asked for minsma=0, extract special isophote there
        if minsma == 0.0:
            # isophote is appended to isophote_list
            _ = self.fit_isophote(0.0, isophote_list=isophote_list)

        # sort list of isophotes according to sma
        isophote_list.sort()

        return IsophoteList(isophote_list)

    def fit_isophote(self, sma, step=0.1, conver=DEFAULT_CONVERGENCE,
                     minit=DEFAULT_MINIT, maxit=DEFAULT_MAXIT,
                     fflag=DEFAULT_FFLAG, maxgerr=DEFAULT_MAXGERR,
                     sclip=3.0, nclip=0, integrmode=BILINEAR,
                     linear=False, maxrit=None, noniterate=False,
                     going_inwards=False, isophote_list=None):
        
        """ Fit a single isophote to the image

        Returns:
            _type_: _description_
        """
        geometry = self._geometry

        # if available, geometry from last fitted isophote will be
        # used as initial guess for next isophote.
        if isophote_list:
            geometry = isophote_list[-1].sample.geometry

        # do the fit
        if noniterate or (maxrit and sma > maxrit):
            isophote = self._non_iterative(sma, step, linear, geometry,
                                           sclip, nclip, integrmode)
        else:
            isophote = self._iterative(sma, step, linear, geometry, sclip,
                                       nclip, integrmode, conver, minit,
                                       maxit, fflag, maxgerr, going_inwards)

        # store result in list
        if isophote_list is not None and isophote.valid:
            isophote_list.append(isophote)

        return isophote

    def _iterative(self, sma, step, linear, geometry, sclip, nclip,
                   integrmode, conver, minit, maxit, fflag, maxgerr,
                   going_inwards=False):
        if sma > 0.0:
            # iterative fitter
            sample = EllipseSample(self.image, sma, astep=step, sclip=sclip,
                                   nclip=nclip, linear_growth=linear,
                                   geometry=geometry, integrmode=integrmode)
            fitter = EllipseFitter(sample)
        else:
            # sma == 0 requires special handling
            sample = CentralEllipseSample(self.image, 0.0, geometry=geometry)
            fitter = CentralEllipseFitter(sample)

        isophote = fitter.fit(conver, minit, maxit, fflag, maxgerr,
                              going_inwards)

        return isophote

    def _non_iterative(self, sma, step, linear, geometry, sclip, nclip,
                       integrmode):
        sample = EllipseSample(self.image, sma, astep=step, sclip=sclip,
                               nclip=nclip, linear_growth=linear,
                               geometry=geometry, integrmode=integrmode)
        sample.update(geometry.fix)

        # build isophote without iterating with an EllipseFitter
        isophote = Isophote(sample, 0, True, stop_code=4)

        return isophote

    @staticmethod
    def _fix_last_isophote(isophote_list, index):
        if isophote_list:
            isophote = isophote_list.pop()

            # check if isophote is bad; if so, fix its geometry
            # to be like the geometry of the index-th isophote
            # in list.
            isophote.fix_geometry(isophote_list[index])

            # force new extraction of raw data, since
            # geometry changed.
            isophote.sample.values = None
            isophote.sample.update(isophote.sample.geometry.fix)

            # we take the opportunity to change an eventual
            # negative stop code to its' positive equivalent.
            code = 5 if isophote.stop_code < 0 else isophote.stop_code

            # build new instance so it can have its attributes
            # populated from the updated sample attributes.
            new_isophote = Isophote(isophote.sample, isophote.niter,
                                    isophote.valid, code)

            # add new isophote to list
            isophote_list.append(new_isophote)
