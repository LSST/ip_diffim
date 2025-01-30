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

import numpy as np

from . import diffimLib
import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.utils.logging import getTraceLogger
from lsst.utils.timer import timeMethod
from .makeKernelBasisList import makeKernelBasisList
from .psfMatch import PsfMatchTask, PsfMatchConfigAL
from . import utils as dituils

__all__ = ("ModelPsfMatchTask", "ModelPsfMatchConfig")

sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))


def nextOddInteger(x):
    nextInt = int(np.ceil(x))
    return nextInt + 1 if nextInt%2 == 0 else nextInt


class ModelPsfMatchConfig(pexConfig.Config):
    """Configuration for model-to-model Psf matching"""

    kernel = pexConfig.ConfigChoiceField(
        doc="kernel type",
        typemap=dict(
            AL=PsfMatchConfigAL,
        ),
        default="AL",
    )
    doAutoPadPsf = pexConfig.Field(
        dtype=bool,
        doc=("If too small, automatically pad the science Psf? "
             "Pad to smallest dimensions appropriate for the matching kernel dimensions, "
             "as specified by autoPadPsfTo. If false, pad by the padPsfBy config."),
        default=True,
    )
    autoPadPsfTo = pexConfig.RangeField(
        dtype=float,
        doc=("Minimum Science Psf dimensions as a fraction of matching kernel dimensions. "
             "If the dimensions of the Psf to be matched are less than the "
             "matching kernel dimensions * autoPadPsfTo, pad Science Psf to this size. "
             "Ignored if doAutoPadPsf=False."),
        default=1.4,
        min=1.0,
        max=2.0
    )
    padPsfBy = pexConfig.Field(
        dtype=int,
        doc="Pixels (even) to pad Science Psf by before matching. Ignored if doAutoPadPsf=True",
        default=0,
    )

    def setDefaults(self):
        # No sigma clipping
        self.kernel.active.singleKernelClipping = False
        self.kernel.active.kernelSumClipping = False
        self.kernel.active.spatialKernelClipping = False
        self.kernel.active.checkConditionNumber = False

        # Variance is ill defined
        self.kernel.active.constantVarianceWeighting = True

        # Do not change specified kernel size
        self.kernel.active.scaleByFwhm = False


class ModelPsfMatchTask(PsfMatchTask):
    """Match two model Psfs, and application of the Psf-matching kernel
    to an input Exposure.
    """
    ConfigClass = ModelPsfMatchConfig

    def __init__(self, *args, **kwargs):
        PsfMatchTask.__init__(self, *args, **kwargs)
        self.kConfig = self.config.kernel.active

    @timeMethod
    def run(self, exposure, referencePsfModel, kernelSum=1.0):
        """Psf-match an exposure to a model Psf.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to Psf-match to the reference Psf model;
            it must return a valid PSF model via exposure.getPsf()
        referencePsfModel : `lsst.afw.detection.Psf`
            The Psf model to match to
        kernelSum : `float`, optional
            A multipicative factor to apply to the kernel sum (default=1.0)

        Returns
        -------
        result : `struct`
            - ``psfMatchedExposure`` : the Psf-matched Exposure.
                This has the same parent bbox, Wcs, PhotoCalib and
                Filter as the input Exposure but no Psf.
                In theory the Psf should equal referencePsfModel but
                the match is likely not exact.
            - ``psfMatchingKernel`` : the spatially varying Psf-matching kernel
            - ``kernelCellSet`` : SpatialCellSet used to solve for the Psf-matching kernel
            - ``referencePsfModel`` : Validated and/or modified reference model used

        Raises
        ------
        RuntimeError
            if the Exposure does not contain a Psf model
        """
        if not exposure.hasPsf():
            raise RuntimeError("exposure does not contain a Psf model")

        maskedImage = exposure.getMaskedImage()

        self.log.info("compute Psf-matching kernel")
        result = self._buildCellSet(exposure, referencePsfModel)
        kernelCellSet = result.kernelCellSet
        referencePsfModel = result.referencePsfModel
        # TODO: This should be evaluated at (or close to) the center of the
        # exposure's bounding box in DM-32756.
        sciAvgPos = exposure.getPsf().getAveragePosition()
        modelAvgPos = referencePsfModel.getAveragePosition()
        fwhmScience = exposure.getPsf().computeShape(sciAvgPos).getDeterminantRadius()*sigma2fwhm
        fwhmModel = referencePsfModel.computeShape(modelAvgPos).getDeterminantRadius()*sigma2fwhm

        basisList = makeKernelBasisList(self.kConfig, fwhmScience, fwhmModel, metadata=self.metadata)
        spatialSolution, psfMatchingKernel, backgroundModel = self._solve(kernelCellSet, basisList)

        if psfMatchingKernel.isSpatiallyVarying():
            sParameters = np.array(psfMatchingKernel.getSpatialParameters())
            sParameters[0][0] = kernelSum
            psfMatchingKernel.setSpatialParameters(sParameters)
        else:
            kParameters = np.array(psfMatchingKernel.getKernelParameters())
            kParameters[0] = kernelSum
            psfMatchingKernel.setKernelParameters(kParameters)

        self.log.info("Psf-match science exposure to reference")
        psfMatchedExposure = afwImage.ExposureF(exposure.getBBox(), exposure.getWcs())
        psfMatchedExposure.info.id = exposure.info.id
        psfMatchedExposure.setFilter(exposure.getFilter())
        psfMatchedExposure.setPhotoCalib(exposure.getPhotoCalib())
        psfMatchedExposure.getInfo().setVisitInfo(exposure.getInfo().getVisitInfo())
        psfMatchedExposure.setPsf(referencePsfModel)
        psfMatchedMaskedImage = psfMatchedExposure.getMaskedImage()

        # Normalize the psf-matching kernel while convolving since its magnitude is meaningless
        # when PSF-matching one model to another.
        convolutionControl = afwMath.ConvolutionControl()
        convolutionControl.setDoNormalize(True)
        afwMath.convolve(psfMatchedMaskedImage, maskedImage, psfMatchingKernel, convolutionControl)

        self.log.info("done")
        return pipeBase.Struct(psfMatchedExposure=psfMatchedExposure,
                               psfMatchingKernel=psfMatchingKernel,
                               kernelCellSet=kernelCellSet,
                               metadata=self.metadata,
                               )

    def _diagnostic(self, kernelCellSet, spatialSolution, spatialKernel, spatialBg):
        """Print diagnostic information on spatial kernel and background fit

        The debugging diagnostics are not really useful here, since the images we are matching have
        no variance.  Thus override the _diagnostic method to generate no logging information"""
        return

    def _buildCellSet(self, exposure, referencePsfModel):
        """Build a SpatialCellSet for use with the solve method

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            The science exposure that will be convolved; must contain a Psf
        referencePsfModel : `lsst.afw.detection.Psf`
            Psf model to match to

        Returns
        -------
        result : `struct`
            - ``kernelCellSet`` : a SpatialCellSet to be used by self._solve
            - ``referencePsfModel`` : Validated and/or modified
                reference model used to populate the SpatialCellSet

        Notes
        -----
        If the reference Psf model and science Psf model have different dimensions,
        adjust the referencePsfModel (the model to which the exposure PSF will be matched)
        to match that of the science Psf. If the science Psf dimensions vary across the image,
        as is common with a WarpedPsf, either pad or clip (depending on config.padPsf)
        the dimensions to be constant.
        """
        sizeCellX = self.kConfig.sizeCellX
        sizeCellY = self.kConfig.sizeCellY

        scienceBBox = exposure.getBBox()
        # Extend for proper spatial matching kernel all the way to edge, especially for narrow strips
        scienceBBox.grow(geom.Extent2I(sizeCellX, sizeCellY))

        sciencePsfModel = exposure.getPsf()

        dimenR = referencePsfModel.getLocalKernel(scienceBBox.getCenter()).getDimensions()

        regionSizeX, regionSizeY = scienceBBox.getDimensions()
        scienceX0, scienceY0 = scienceBBox.getMin()

        kernelCellSet = afwMath.SpatialCellSet(geom.Box2I(scienceBBox), sizeCellX, sizeCellY)

        nCellX = regionSizeX//sizeCellX
        nCellY = regionSizeY//sizeCellY

        if nCellX == 0 or nCellY == 0:
            raise ValueError("Exposure dimensions=%s and sizeCell=(%s, %s). Insufficient area to match" %
                             (scienceBBox.getDimensions(), sizeCellX, sizeCellY))

        # Survey the PSF dimensions of the Spatial Cell Set
        # to identify the minimum enclosed or maximum bounding square BBox.
        widthList = []
        heightList = []
        for row in range(nCellY):
            posY = sizeCellY*row + sizeCellY//2 + scienceY0
            for col in range(nCellX):
                posX = sizeCellX*col + sizeCellX//2 + scienceX0
                widthS, heightS = sciencePsfModel.computeBBox(geom.Point2D(posX, posY)).getDimensions()
                widthList.append(widthS)
                heightList.append(heightS)

        psfSize = max(max(heightList), max(widthList))

        if self.config.doAutoPadPsf:
            minPsfSize = nextOddInteger(self.kConfig.kernelSize*self.config.autoPadPsfTo)
            paddingPix = max(0, minPsfSize - psfSize)
        else:
            if self.config.padPsfBy % 2 != 0:
                raise ValueError("Config padPsfBy (%i pixels) must be even number." %
                                 self.config.padPsfBy)
            paddingPix = self.config.padPsfBy

        if paddingPix > 0:
            self.log.debug("Padding Science PSF from (%d, %d) to (%d, %d) pixels",
                           psfSize, psfSize, paddingPix + psfSize, paddingPix + psfSize)
            psfSize += paddingPix

        # Check that PSF is larger than the matching kernel
        maxKernelSize = psfSize - 1
        if maxKernelSize % 2 == 0:
            maxKernelSize -= 1
        if self.kConfig.kernelSize > maxKernelSize:
            message = """
                Kernel size (%d) too big to match Psfs of size %d.
                Please reconfigure by setting one of the following:
                1) kernel size to <= %d
                2) doAutoPadPsf=True
                3) padPsfBy to >= %s
                """ % (self.kConfig.kernelSize, psfSize,
                       maxKernelSize, self.kConfig.kernelSize - maxKernelSize)
            raise ValueError(message)

        dimenS = geom.Extent2I(psfSize, psfSize)

        if (dimenR != dimenS):
            try:
                referencePsfModel = referencePsfModel.resized(psfSize, psfSize)
                self.log.info("Adjusted dimensions of reference PSF model from %s to %s", dimenR, dimenS)
            except Exception as e:
                self.log.warning("Zero padding or clipping the reference PSF model of type %s and dimensions"
                                 " %s to the science Psf dimensions %s because: %s",
                                 referencePsfModel.__class__.__name__, dimenR, dimenS, e)
            dimenR = dimenS

        ps = pexConfig.makePropertySet(self.kConfig)
        if self.kConfig.kernelBasisSet == "combined":
            for field in set(dir(self.kConfig.psfMatchDF.value)) - set(dir(self.kConfig)):
                ps.add(field, getattr(self.kConfig.psfMatchDF.value, field))

        for row in range(nCellY):
            # place at center of cell
            posY = sizeCellY*row + sizeCellY//2 + scienceY0

            for col in range(nCellX):
                # place at center of cell
                posX = sizeCellX*col + sizeCellX//2 + scienceX0

                getTraceLogger(self.log, 4).debug("Creating Psf candidate at %.1f %.1f", posX, posY)

                # reference kernel image, at location of science subimage
                referenceMI = self._makePsfMaskedImage(referencePsfModel, posX, posY, dimensions=dimenR)

                # kernel image we are going to convolve
                scienceMI = self._makePsfMaskedImage(sciencePsfModel, posX, posY, dimensions=dimenR)

                # The image to convolve is the science image, to the reference Psf.
                kc = diffimLib.makeKernelCandidate(posX, posY, scienceMI, referenceMI, ps)
                kernelCellSet.insertCandidate(kc)

        import lsstDebug
        display = lsstDebug.Info(__name__).display
        displaySpatialCells = lsstDebug.Info(__name__).displaySpatialCells
        maskTransparency = lsstDebug.Info(__name__).maskTransparency
        if not maskTransparency:
            maskTransparency = 0
        if display:
            afwDisplay.setDefaultMaskTransparency(maskTransparency)
        if display and displaySpatialCells:
            dituils.showKernelSpatialCells(exposure.getMaskedImage(), kernelCellSet,
                                           symb="o", ctype=afwDisplay.CYAN, ctypeUnused=afwDisplay.YELLOW,
                                           ctypeBad=afwDisplay.RED, size=4, frame=lsstDebug.frame,
                                           title="Image to be convolved")
            lsstDebug.frame += 1
        return pipeBase.Struct(kernelCellSet=kernelCellSet,
                               referencePsfModel=referencePsfModel,
                               )

    def _makePsfMaskedImage(self, psfModel, posX, posY, dimensions=None):
        """Return a MaskedImage of the a PSF Model of specified dimensions
        """
        rawKernel = psfModel.computeKernelImage(geom.Point2D(posX, posY)).convertF()
        if dimensions is None:
            dimensions = rawKernel.getDimensions()
        if rawKernel.getDimensions() == dimensions:
            kernelIm = rawKernel
        else:
            # make image of proper size
            kernelIm = afwImage.ImageF(dimensions)
            bboxToPlace = geom.Box2I(geom.Point2I((dimensions.getX() - rawKernel.getWidth())//2,
                                                  (dimensions.getY() - rawKernel.getHeight())//2),
                                     rawKernel.getDimensions())
            kernelIm.assign(rawKernel, bboxToPlace)

        kernelMask = afwImage.Mask(dimensions, 0x0)
        kernelVar = afwImage.ImageF(dimensions, 1.0)
        return afwImage.MaskedImageF(kernelIm, kernelMask, kernelVar)
