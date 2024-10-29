# This file is part of ip_diffim.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Support utilities for Measuring sources"""


__all__ = ["evaluateMeanPsfFwhm", "getPsfFwhm", "getKernelCenterDisplacement",
           "divideExposureByPatches"]

import itertools
import numpy as np
import lsst.geom as geom
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.meas.algorithms import SpatialCellCandidate
from lsst.pex.exceptions import InvalidParameterError, RangeError
from lsst.utils.logging import getLogger


_LOG = getLogger(__name__)


def getKernelCenterDisplacement(kernel, x, y, image=None):
    """Calculate the PSF matching kernel peak offset from the nominal
    position.

    Parameters
    ----------
    kernel : `~lsst.afw.math.LinearCombinationKernel`
        The PSF matching kernel to evaluate.
    x : `float`
        The x position on the detector to evaluate the kernel
    y : `float`
        The y position on the detector to evaluate the kernel
    image : `~lsst.afw.image.ImageD`
        The image to use as base for computing kernel pixel values

    Returns
    -------
    kernel_sum : `float`
        The sum of the kernel on the desired location
    dx : `float`
        The displacement of the kernel averaged peak, with respect to the
        center of the extraction of the kernel
    dy : `float`
        The displacement of the kernel averaged peak, with respect to the
        center of the extraction of the kernel
    pos_angle: `float`
        The position angle in detector coordinates of the displacement
    length : `float`
        The displacement module of the kernel centroid in pixel units
    """

    if image is None:
        image = afwImage.ImageD(kernel.getDimensions())

    # obtain the kernel image
    hsize = kernel.getWidth()//2
    kernel_sum = kernel.computeImage(image, doNormalize=False, x=x, y=y)

    data = image.array
    h, w = data.shape
    xx = np.arange(w)
    yy = np.arange(h)

    # create sum vectors and estimate weighted average
    vx = data.sum(axis=0)
    vx /= vx.sum()
    dx = np.dot(vx, xx) - hsize

    vy = data.sum(axis=1)
    vy /= vy.sum()
    dy = np.dot(vy, yy) - hsize

    # obtain position angle and norm of displacement
    pos_angle = np.arctan2(dy, dx)
    length = np.sqrt(dx**2 + dy**2)

    return kernel_sum, dx, dy, pos_angle, length


def getPsfFwhm(psf, average=True, position=None):
    """Directly calculate the horizontal and vertical widths
    of a PSF at half its maximum value.

    Parameters
    ----------
    psf : `~lsst.afw.detection.Psf`
        Point spread function (PSF) to evaluate.
    average : `bool`, optional
        Set to return the average width over Y and X axes.
    position : `~lsst.geom.Point2D`, optional
        The position at which to evaluate the PSF. If `None`, then the
        average position is used.

    Returns
    -------
    psfSize : `float` | `tuple` [`float`]
        The FWHM of the PSF computed at its average position.
        Returns the widths along the Y and X axes,
        or the average of the two if `average` is set.

    See Also
    --------
    evaluateMeanPsfFwhm
    """
    if position is None:
        position = psf.getAveragePosition()
    shape = psf.computeShape(position)
    sigmaToFwhm = 2*np.log(2*np.sqrt(2))

    if average:
        return sigmaToFwhm*shape.getTraceRadius()
    else:
        return [sigmaToFwhm*np.sqrt(shape.getIxx()), sigmaToFwhm*np.sqrt(shape.getIyy())]


def evaluateMeanPsfFwhm(exposure: afwImage.Exposure,
                        fwhmExposureBuffer: float, fwhmExposureGrid: int) -> float:
    """Get the mean PSF FWHM by evaluating it on a grid within an exposure.

    Parameters
    ----------
    exposure : `~lsst.afw.image.Exposure`
        The exposure for which the mean FWHM of the PSF is to be computed.
        The exposure must contain a `psf` attribute.
    fwhmExposureBuffer : `float`
        Fractional buffer margin to be left out of all sides of the image
        during the construction of the grid to compute mean PSF FWHM in an
        exposure.
    fwhmExposureGrid : `int`
        Grid size to compute the mean FWHM in an exposure.

    Returns
    -------
    meanFwhm : `float`
        The mean PSF FWHM on the exposure.

    Raises
    ------
    ValueError
        Raised if the PSF cannot be computed at any of the grid points.

    See Also
    --------
    `getPsfFwhm`
    `computeAveragePsf`
    """

    psf = exposure.psf

    bbox = exposure.getBBox()
    xmax, ymax = bbox.getMax()
    xmin, ymin = bbox.getMin()

    xbuffer = fwhmExposureBuffer*(xmax-xmin)
    ybuffer = fwhmExposureBuffer*(ymax-ymin)

    width = []
    for (x, y) in itertools.product(np.linspace(xmin+xbuffer, xmax-xbuffer, fwhmExposureGrid),
                                    np.linspace(ymin+ybuffer, ymax-ybuffer, fwhmExposureGrid)
                                    ):
        pos = geom.Point2D(x, y)
        try:
            fwhm = getPsfFwhm(psf, average=True, position=pos)
        except (InvalidParameterError, RangeError):
            _LOG.debug("Unable to compute PSF FWHM at position (%f, %f).", x, y)
            continue

        width.append(fwhm)

    if not width:
        raise ValueError("Unable to compute PSF FWHM at any position on the exposure.")

    return np.nanmean(width)


def computeAveragePsf(exposure: afwImage.Exposure,
                      psfExposureBuffer: float, psfExposureGrid: int) -> afwImage.ImageD:
    """Get the average PSF by evaluating it on a grid within an exposure.

    Parameters
    ----------
    exposure : `~lsst.afw.image.Exposure`
        The exposure for which the average PSF is to be computed.
        The exposure must contain a `psf` attribute.
    psfExposureBuffer : `float`
        Fractional buffer margin to be left out of all sides of the image
        during the construction of the grid to compute average PSF in an
        exposure.
    psfExposureGrid : `int`
        Grid size to compute the average PSF in an exposure.

    Returns
    -------
    psfImage : `~lsst.afw.image.Image`
        The average PSF across the exposure.

    Raises
    ------
    ValueError
        Raised if the PSF cannot be computed at any of the grid points.

    See Also
    --------
    `evaluateMeanPsfFwhm`
    """

    psf = exposure.psf

    bbox = exposure.getBBox()
    xmax, ymax = bbox.getMax()
    xmin, ymin = bbox.getMin()

    xbuffer = psfExposureBuffer*(xmax-xmin)
    ybuffer = psfExposureBuffer*(ymax-ymin)

    nImg = 0
    psfArray = None
    for (x, y) in itertools.product(np.linspace(xmin+xbuffer, xmax-xbuffer, psfExposureGrid),
                                    np.linspace(ymin+ybuffer, ymax-ybuffer, psfExposureGrid)
                                    ):
        pos = geom.Point2D(x, y)
        try:
            singleImage = psf.computeKernelImage(pos)
        except InvalidParameterError:
            _LOG.debug("Unable to compute PSF image at position (%f, %f).", x, y)
            continue

        if psfArray is None:
            psfArray = singleImage.array
        else:
            psfArray += singleImage.array
        nImg += 1

    if psfArray is None:
        raise ValueError("Unable to compute PSF image at any position on the exposure.")

    psfImage = afwImage.ImageD(psfArray/nImg)
    return psfImage


def computeRobustStatistics(image, mask, statsCtrl, statistic=afwMath.MEANCLIP):
    """Calculate a robust mean of the variance plane of an exposure.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image or variance plane of an exposure to evaluate.
    mask : `lsst.afw.image.Mask`
        Mask plane to use for excluding pixels.
    statsCtrl : `lsst.afw.math.StatisticsControl`
        Statistics control object for configuring the calculation.
    statistic : `lsst.afw.math.Property`, optional
        The type of statistic to compute. Typical values are
        ``afwMath.MEANCLIP`` or ``afwMath.STDEVCLIP``.

    Returns
    -------
    value : `float`
        The result of the statistic calculated from the unflagged pixels.
    """
    statObj = afwMath.makeStatistics(image, mask, statistic, statsCtrl)
    return statObj.getValue(statistic)


def computePSFNoiseEquivalentArea(psf):
    """Compute the noise equivalent area for an image psf

    Parameters
    ----------
    psf : `lsst.afw.detection.Psf`

    Returns
    -------
    nea : `float`
    """
    psfImg = psf.computeImage(psf.getAveragePosition())
    nea = 1./np.sum(psfImg.array**2)
    return nea


def angleMean(angles):
    """Calculate the mean of an array of angles.

    Parameters
    ----------
    angles : `ndarray`
        An array of angles, in radians

    Returns
    -------
    `lsst.geom.Angle`
        The mean angle
    """
    complexArray = [complex(np.cos(angle), np.sin(angle)) for angle in angles]
    return (geom.Angle(np.angle(np.mean(complexArray))))


def evaluateMaskFraction(mask, maskPlane):
    """Evaluate the fraction of masked pixels in a mask plane.

    Parameters
    ----------
    mask : `lsst.afw.image.Mask`
        The mask to evaluate the fraction on
    maskPlane : `str`
        The particular mask plane to evaluate the fraction

    Returns
    -------
    value : `float`
        The calculated fraction of masked pixels
    """
    nMaskSet = np.count_nonzero((mask.array & mask.getPlaneBitMask(maskPlane)))
    return nMaskSet/mask.array.size


def divideExposureByPatches(exposure, skymap, overlapThreshold=0.1):
    """Summary

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Description
    skymap : `lsst.skymap.SkyMap`
        Description
    overlapThreshold : `float`, optional
        Description

    Returns
    -------
    patchCandidates : `list` of `lsst.skymap.PatchInfo`
        Description
    """
    corners = exposure.wcs.pixelToSky(geom.Box2D(exposure.getBBox()).getCorners())
    tractList = skymap.findTractPatchList(corners)
    detectorPolygon = geom.Box2D(exposure.getBBox())
    area = exposure.getBBox().getArea()
    patchCandidates = []
    overlapFraction = []
    tractCoverageList = []
    for tractInfo, patchList in tractList:
        tractCheck = [tractInfo.contains(exposure.wcs.pixelToSky(corner))
                      for corner in detectorPolygon.getCorners()]
        tractCoverage = 0
        if np.all(tractCheck):
            for patch in patchList:
                patchCorners = patch.wcs.pixelToSky(geom.Box2D(patch.getOuterBBox()).getCorners())
                patchPolygon = afwGeom.Polygon(exposure.wcs.skyToPixel(patchCorners))
                overlappingArea = patchPolygon.intersectionSingle(detectorPolygon).calculateArea()
                if overlappingArea/area >= overlapThreshold*2:
                    tractCoverage += overlappingArea/area
        # Coverage metric can exceed 1 since we are using the outer BBox.
        tractCoverageList.append(np.min([tractCoverage, 1]))
    if np.max(tractCoverageList) > 0:
        print(tractCoverageList)
        tractUse = np.argmax(tractCoverageList)
    else:
        tractContainsList = []
        for tractInfo, patchList in tractList:
            tractCheck = tractInfo.contains(exposure.wcs.pixelToSky(detectorPolygon.getCenter()))
            tractContainsList.append(tractCheck)
        tractUse = np.argmax(tractContainsList)
        
    tractInfo, patchList = tractList.pop(tractUse)
    for patch in patchList:
        # Switch to using the inner BBox
        patchCorners = patch.wcs.pixelToSky(geom.Box2D(patch.getInnerBBox()).getCorners())
        patchPolygon = afwGeom.Polygon(exposure.wcs.skyToPixel(patchCorners))
        if patchPolygon.overlaps(detectorPolygon):
            overlappingArea = patchPolygon.intersectionSingle(detectorPolygon).calculateArea()
        else:
            overlappingArea = 0
        if overlappingArea/area >= overlapThreshold:
            patchCandidates.append(patch)
            overlapFraction.append(overlappingArea/area)
    
    # Sort the patches by overlap fraction, largest last
    patchCandidates = [p[1] for p in sorted(zip(overlapFraction, patchCandidates), reverse=False)]
    return patchCandidates


class MyKernelSpatialCellCandidate(SpatialCellCandidate):
    def __init__(self, region, kernel):
        """Initialize with position, region, and kernel.

        Parameters:
        x, y -- coordinates of the candidate (e.g., centroid of the region)
        region -- lsst.afw.geom.Polygon defining the valid region for the kernel
        kernel -- lsst.afw.math.LinearCombinationKernel for this region
        """
        super().__init__()
        self.region = region
        self.kernel = kernel

    def getKernel(self):
        """Return the kernel associated with this candidate."""
        return self.kernel

    def getRegion(self):
        """Return the region (Polygon) associated with this candidate."""
        return self.region

    def isInRegion(self, point):
        """Check if a given point (lsst.geom.Point2D) is inside the region."""
        return self.region.contains(point)

    def getChi2(self):
        """Return a dummy chi-squared value for sorting candidates (if needed)."""
        return 0.0
