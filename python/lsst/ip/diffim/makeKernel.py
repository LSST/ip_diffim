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

__all__ = ["MakeKernelConfig", "MakeKernelTask"]

import numpy as np

import lsst.afw.detection
import lsst.afw.image
import lsst.afw.math
import lsst.afw.table
import lsst.daf.base
import lsst.geom
from lsst.meas.algorithms import SourceDetectionTask, SubtractBackgroundTask
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.pex.exceptions import InvalidParameterError, RangeError
import lsst.pex.config
import lsst.pipe.base

from .makeKernelBasisList import makeKernelBasisList
from .psfMatch import PsfMatchConfig, PsfMatchTask, PsfMatchConfigAL, PsfMatchConfigDF

from . import diffimLib
from .utils import evaluateMeanPsfFwhm, getPsfFwhm


class MakeKernelConfig(PsfMatchConfig):
    kernel = lsst.pex.config.ConfigChoiceField(
        doc="kernel type",
        typemap=dict(
            AL=PsfMatchConfigAL,
            DF=PsfMatchConfigDF
        ),
        default="AL",
    )
    selectDetection = lsst.pex.config.ConfigurableField(
        target=SourceDetectionTask,
        doc="Initial detections used to feed stars to kernel fitting",
    )
    selectMeasurement = lsst.pex.config.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Initial measurements used to feed stars to kernel fitting",
    )
    fwhmExposureGrid = lsst.pex.config.Field(
        doc="Grid size to compute the average PSF FWHM in an exposure",
        dtype=int,
        default=10,
    )
    fwhmExposureBuffer = lsst.pex.config.Field(
        doc="Fractional buffer margin to be left out of all sides of the image during construction"
            "of grid to compute average PSF FWHM in an exposure",
        dtype=float,
        default=0.05,
    )

    def setDefaults(self):
        # High sigma detections only
        self.selectDetection.reEstimateBackground = False
        self.selectDetection.thresholdValue = 10.0
        self.selectDetection.excludeMaskPlanes = ["EDGE"]

        # Minimal set of measurments for star selection
        self.selectMeasurement.algorithms.names.clear()
        self.selectMeasurement.algorithms.names = ('base_SdssCentroid', 'base_PsfFlux', 'base_PixelFlags',
                                                   'base_SdssShape', 'base_GaussianFlux', 'base_SkyCoord')
        self.selectMeasurement.slots.modelFlux = None
        self.selectMeasurement.slots.apFlux = None
        self.selectMeasurement.slots.calibFlux = None


class MakeKernelTask(PsfMatchTask):
    """Construct a kernel for PSF matching two exposures.
    """

    ConfigClass = MakeKernelConfig
    _DefaultName = "makeALKernel"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the background subtraction task uses a config from an unusual location,
        # so cannot easily be constructed with makeSubtask
        self.background = SubtractBackgroundTask(config=self.kConfig.afwBackgroundConfig, name="background",
                                                 parentTask=self)

        self.selectSchema = lsst.afw.table.SourceTable.makeMinimalSchema()
        # Add coordinate error fields:
        lsst.afw.table.CoordKey.addErrorFields(self.selectSchema)
        self.selectAlgMetadata = lsst.daf.base.PropertyList()
        self.makeSubtask("selectDetection", schema=self.selectSchema)
        self.makeSubtask("selectMeasurement", schema=self.selectSchema, algMetadata=self.selectAlgMetadata)

    def run(self, template, science, kernelSources, preconvolved=False):
        """Solve for the kernel and background model that best match two
        Exposures evaluated at the given source locations.

        Parameters
        ----------
        template : `lsst.afw.image.Exposure`
            Exposure that will be convolved.
        science : `lsst.afw.image.Exposure`
            The exposure that will be matched.
        kernelSources : `lsst.afw.table.SourceCatalog`
            Kernel candidate sources with appropriately sized footprints.
            Typically the output of `MakeKernelTask.selectKernelSources`.
        preconvolved : `bool`, optional
            Was the science image convolved with its own PSF?

        Returns
        -------
        results : `lsst.pipe.base.Struct`

            ``psfMatchingKernel`` : `lsst.afw.math.LinearCombinationKernel`
                Spatially varying Psf-matching kernel.
            ``backgroundModel``  : `lsst.afw.math.Function2D`
                Spatially varying background-matching function.
        """
        kernelCellSet = self._buildCellSet(template.maskedImage, science.maskedImage, kernelSources)
        # Calling getPsfFwhm on template.psf fails on some rare occasions when
        # the template has no input exposures at the average position of the
        # stars. So we try getPsfFwhm first on template, and if that fails we
        # evaluate the PSF on a grid specified by fwhmExposure* fields.
        # To keep consistent definitions for PSF size on the template and
        # science images, we use the same method for both.
        try:
            templateFwhmPix = getPsfFwhm(template.psf)
            scienceFwhmPix = getPsfFwhm(science.psf)
        except (InvalidParameterError, RangeError):
            self.log.debug("Unable to evaluate PSF at the average position. "
                           "Evaluting PSF on a grid of points."
                           )
            templateFwhmPix = evaluateMeanPsfFwhm(template,
                                                  fwhmExposureBuffer=self.config.fwhmExposureBuffer,
                                                  fwhmExposureGrid=self.config.fwhmExposureGrid
                                                  )
            scienceFwhmPix = evaluateMeanPsfFwhm(science,
                                                 fwhmExposureBuffer=self.config.fwhmExposureBuffer,
                                                 fwhmExposureGrid=self.config.fwhmExposureGrid
                                                 )

        if preconvolved:
            scienceFwhmPix *= np.sqrt(2)
        basisList = self.makeKernelBasisList(templateFwhmPix, scienceFwhmPix,
                                             metadata=self.metadata)
        spatialSolution, psfMatchingKernel, backgroundModel = self._solve(kernelCellSet, basisList)
        return lsst.pipe.base.Struct(
            psfMatchingKernel=psfMatchingKernel,
            backgroundModel=backgroundModel,
        )

    def selectKernelSources(self, template, science, candidateList=None, preconvolved=False):
        """Select sources from a list of candidates, and extract footprints.

        Parameters
        ----------
        template : `lsst.afw.image.Exposure`
            Exposure that will be convolved.
        science : `lsst.afw.image.Exposure`
            The exposure that will be matched.
        candidateList : `lsst.afw.table.SourceCatalog`
            Sources to check as possible kernel candidates.
        preconvolved : `bool`, optional
            Was the science image convolved with its own PSF?

        Returns
        -------
        kernelSources : `lsst.afw.table.SourceCatalog`
            Kernel candidates with appropriate sized footprints.
        """
        # Calling getPsfFwhm on template.psf fails on some rare occasions when
        # the template has no input exposures at the average position of the
        # stars. So we try getPsfFwhm first on template, and if that fails we
        # evaluate the PSF on a grid specified by fwhmExposure* fields.
        # To keep consistent definitions for PSF size on the template and
        # science images, we use the same method for both.
        try:
            templateFwhmPix = getPsfFwhm(template.psf)
            scienceFwhmPix = getPsfFwhm(science.psf)
        except (InvalidParameterError, RangeError):
            self.log.debug("Unable to evaluate PSF at the average position. "
                           "Evaluting PSF on a grid of points."
                           )
            templateFwhmPix = evaluateMeanPsfFwhm(template,
                                                  fwhmExposureBuffer=self.config.fwhmExposureBuffer,
                                                  fwhmExposureGrid=self.config.fwhmExposureGrid
                                                  )
            scienceFwhmPix = evaluateMeanPsfFwhm(science,
                                                 fwhmExposureBuffer=self.config.fwhmExposureBuffer,
                                                 fwhmExposureGrid=self.config.fwhmExposureGrid
                                                 )
        if preconvolved:
            scienceFwhmPix *= np.sqrt(2)
        kernelSize = self.makeKernelBasisList(templateFwhmPix, scienceFwhmPix)[0].getWidth()
        kernelSources = self.makeCandidateList(template, science, kernelSize,
                                               candidateList=candidateList,
                                               preconvolved=preconvolved)
        return kernelSources

    def getSelectSources(self, exposure, sigma=None, doSmooth=True, idFactory=None):
        """Get sources to use for Psf-matching.

        This method runs detection and measurement on an exposure.
        The returned set of sources will be used as candidates for
        Psf-matching.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure on which to run detection/measurement
        sigma : `float`, optional
            PSF sigma, in pixels, used for smoothing the image for detection.
            If `None`, the PSF width will be used.
        doSmooth : `bool`
            Whether or not to smooth the Exposure with Psf before detection
        idFactory : `lsst.afw.table.IdFactory`
            Factory for the generation of Source ids

        Returns
        -------
        selectSources :
            source catalog containing candidates for the Psf-matching
        """
        if idFactory:
            table = lsst.afw.table.SourceTable.make(self.selectSchema, idFactory)
        else:
            table = lsst.afw.table.SourceTable.make(self.selectSchema)
        mi = exposure.getMaskedImage()

        imArr = mi.image.array
        maskArr = mi.mask.array
        miArr = np.ma.masked_array(imArr, mask=maskArr)
        try:
            fitBg = self.background.fitBackground(mi)
            bkgd = fitBg.getImageF(self.background.config.algorithm,
                                   self.background.config.undersampleStyle)
        except Exception:
            self.log.warning("Failed to get background model. Falling back to median background estimation")
            bkgd = np.ma.median(miArr)

        # Take off background for detection
        mi -= bkgd
        try:
            table.setMetadata(self.selectAlgMetadata)
            detRet = self.selectDetection.run(
                table=table,
                exposure=exposure,
                sigma=sigma,
                doSmooth=doSmooth
            )
            selectSources = detRet.sources
            self.selectMeasurement.run(measCat=selectSources, exposure=exposure)
        finally:
            # Put back on the background in case it is needed down stream
            mi += bkgd
            del bkgd

        self.log.info("Selected %d sources via detection measurement.", len(selectSources))
        return selectSources

    def makeCandidateList(self, convolved, reference, kernelSize,
                          candidateList, preconvolved=False, sigma=None):
        """Make a list of acceptable KernelCandidates.

        Generate a list of candidate sources for Psf-matching, remove sources
        with bad pixel masks set or that extend off the image.

        Parameters
        ----------
        convolved : `lsst.afw.image.Exposure`
            Exposure that will be convolved. This is typically the template
            image, and may have a large bbox than the reference exposure.
        reference : `lsst.afw.image.Exposure`
            Exposure that will be matched-to. This is typically the science
            image.
        kernelSize : `float`
            Dimensions of the Psf-matching Kernel, used to set detection
            footprints.
        candidateList : `lsst.afw.table.SourceCatalog`
            List of Sources to examine for kernel candidacy.
        preconvolved : `bool`, optional
            Was the science exposure already convolved with its PSF?

        Returns
        -------
        candidates : `lsst.afw.table.SourceCatalog`
            Candidates with footprints extended to a ``kernelSize`` box.

        Raises
        ------
        RuntimeError
            If ``candidateList`` is empty after sub-selection.
        """
        if candidateList is None:
            candidateList = self.getSelectSources(reference, doSmooth=not preconvolved, sigma=sigma)
            if len(candidateList) < 1:
                raise RuntimeError("No kernel candidates after detection and measurement.")

        bitmask = reference.mask.getPlaneBitMask(self.config.badMaskPlanes)
        good = np.ones(len(candidateList), dtype=bool)
        # Make all candidates have the same size footprint, based on kernelSize.
        for i, candidate in enumerate(candidateList):
            # Only use the brightest peak; the input should be pre-deblended!
            peak = candidate.getFootprint().getPeaks()[0]
            size = 2*kernelSize + 1  # ensure the resulting box is odd
            bbox = lsst.geom.Box2I.makeCenteredBox(candidate.getCentroid(),
                                                   lsst.geom.Extent2I(size, size))
            boxFootprint = lsst.afw.detection.Footprint(lsst.afw.geom.SpanSet(bbox))
            boxFootprint.addPeak(peak.getFx(), peak.getFy(), peak.getPeakValue())
            candidate.setFootprint(boxFootprint)

            # Reject footprints not contained in either image.
            if not reference.getBBox().contains(bbox) or not convolved.getBBox().contains(bbox):
                good[i] = False
                continue
            # Reject footprints with any bad mask bits set.
            if (reference.subset(bbox).mask.array & bitmask).any():
                good[i] = False
                continue
        candidates = candidateList[good].copy(deep=True)

        self.log.info("Selected %d / %d sources as kernel candidates.", good.sum(), len(candidateList))

        if len(candidates) < 1:
            raise RuntimeError("No good kernel candidates available.")

        return candidates

    def makeKernelBasisList(self, targetFwhmPix=None, referenceFwhmPix=None,
                            basisDegGauss=None, basisSigmaGauss=None, metadata=None):
        """Wrapper to set log messages for
        `lsst.ip.diffim.makeKernelBasisList`.

        Parameters
        ----------
        targetFwhmPix : `float`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.
        referenceFwhmPix : `float`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.
        basisDegGauss : `list` of `int`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.
        basisSigmaGauss : `list` of `int`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.
        metadata : `lsst.daf.base.PropertySet`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.

        Returns
        -------
        basisList: `list` of `lsst.afw.math.kernel.FixedKernel`
            List of basis kernels.
        """
        basisList = makeKernelBasisList(self.kConfig,
                                        targetFwhmPix=targetFwhmPix,
                                        referenceFwhmPix=referenceFwhmPix,
                                        basisDegGauss=basisDegGauss,
                                        basisSigmaGauss=basisSigmaGauss,
                                        metadata=metadata)
        if targetFwhmPix == referenceFwhmPix:
            self.log.info("Target and reference psf fwhms are equal, falling back to config values")
        elif referenceFwhmPix > targetFwhmPix:
            self.log.info("Reference psf fwhm is the greater, normal convolution mode")
        else:
            self.log.info("Target psf fwhm is the greater, deconvolution mode")

        return basisList

    def _buildCellSet(self, convolved, reference, candidateList):
        """Build a SpatialCellSet for use with the solve method.

        Parameters
        ----------
        convolved : `lsst.afw.image.MaskedImage`
            MaskedImage to PSF-matched to reference.
        reference : `lsst.afw.image.MaskedImage`
            Reference MaskedImage.
        candidateList : `lsst.afw.table.SourceCatalog`
            Kernel candidate sources with footprints.

        Returns
        -------
        kernelCellSet : `lsst.afw.math.SpatialCellSet`
            A SpatialCellSet for use with self._solve.
        """
        sizeCellX, sizeCellY = self._adaptCellSize(candidateList)

        imageBBox = convolved.getBBox()
        imageBBox.clip(reference.getBBox())
        # Object to store the KernelCandidates for spatial modeling
        kernelCellSet = lsst.afw.math.SpatialCellSet(imageBBox, sizeCellX, sizeCellY)

        candidateConfig = lsst.pex.config.makePropertySet(self.kConfig)
        # Place candidates within the spatial grid
        for candidate in candidateList:
            bbox = candidate.getFootprint().getBBox()
            templateCutout = lsst.afw.image.MaskedImageF(convolved, bbox)
            scienceCutout = lsst.afw.image.MaskedImageF(reference, bbox)

            kernelCandidate = diffimLib.makeKernelCandidate(candidate,
                                                            templateCutout,
                                                            scienceCutout,
                                                            candidateConfig)

            self.log.debug("Candidate %d at %.2f, %.2f rating=%f",
                           kernelCandidate.getId(),
                           kernelCandidate.getXCenter(),
                           kernelCandidate.getYCenter(),
                           kernelCandidate.getCandidateRating())
            kernelCellSet.insertCandidate(kernelCandidate)

        return kernelCellSet

    def _adaptCellSize(self, candidateList):
        """NOT IMPLEMENTED YET.

        Parameters
        ----------
        candidateList : `list`
            A list of footprints/maskedImages for kernel candidates;

        Returns
        -------
        sizeCellX, sizeCellY : `int`
            New dimensions to use for the kernel.
        """
        return self.kConfig.sizeCellX, self.kConfig.sizeCellY
