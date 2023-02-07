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

import lsst.afw.image
import lsst.afw.math
import lsst.geom
from lsst.ip.diffim.utils import evaluateMeanPsfFwhm, getPsfFwhm
from lsst.meas.algorithms import ScaleVarianceTask
import lsst.pex.config
import lsst.pipe.base
from lsst.pex.exceptions import InvalidParameterError
from lsst.pipe.base import connectionTypes
from . import MakeKernelTask, DecorrelateALKernelTask
from lsst.utils.timer import timeMethod

__all__ = ["AlardLuptonSubtractConfig", "AlardLuptonSubtractTask",
           "AlardLuptonPreconvolveSubtractConfig", "AlardLuptonPreconvolveSubtractTask"]

_dimensions = ("instrument", "visit", "detector")
_defaultTemplates = {"coaddName": "deep", "fakesType": ""}


class SubtractInputConnections(lsst.pipe.base.PipelineTaskConnections,
                               dimensions=_dimensions,
                               defaultTemplates=_defaultTemplates):
    template = connectionTypes.Input(
        doc="Input warped template to subtract.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_templateExp"
    )
    science = connectionTypes.Input(
        doc="Input science exposure to subtract from.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}calexp"
    )
    sources = connectionTypes.Input(
        doc="Sources measured on the science exposure; "
            "used to select sources for making the matching kernel.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        name="{fakesType}src"
    )
    finalizedPsfApCorrCatalog = connectionTypes.Input(
        doc=("Per-visit finalized psf models and aperture correction maps. "
             "These catalogs use the detector id for the catalog id, "
             "sorted on id for fast lookup."),
        dimensions=("instrument", "visit"),
        storageClass="ExposureCatalog",
        name="finalVisitSummary",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doApplyFinalizedPsf:
            self.inputs.remove("finalizedPsfApCorrCatalog")


class SubtractImageOutputConnections(lsst.pipe.base.PipelineTaskConnections,
                                     dimensions=_dimensions,
                                     defaultTemplates=_defaultTemplates):
    difference = connectionTypes.Output(
        doc="Result of subtracting convolved template from science image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_differenceTempExp",
    )
    matchedTemplate = connectionTypes.Output(
        doc="Warped and PSF-matched template used to create `subtractedExposure`.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_matchedExp",
    )


class SubtractScoreOutputConnections(lsst.pipe.base.PipelineTaskConnections,
                                     dimensions=_dimensions,
                                     defaultTemplates=_defaultTemplates):
    scoreExposure = connectionTypes.Output(
        doc="The maximum likelihood image, used for the detection of diaSources.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_scoreExp",
    )


class AlardLuptonSubtractConnections(SubtractInputConnections, SubtractImageOutputConnections):
    pass


class AlardLuptonSubtractBaseConfig(lsst.pex.config.Config):
    makeKernel = lsst.pex.config.ConfigurableField(
        target=MakeKernelTask,
        doc="Task to construct a matching kernel for convolution.",
    )
    doDecorrelation = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Perform diffim decorrelation to undo pixel correlation due to A&L "
        "kernel convolution? If True, also update the diffim PSF."
    )
    decorrelate = lsst.pex.config.ConfigurableField(
        target=DecorrelateALKernelTask,
        doc="Task to decorrelate the image difference.",
    )
    requiredTemplateFraction = lsst.pex.config.Field(
        dtype=float,
        default=0.1,
        doc="Abort task if template covers less than this fraction of pixels."
        " Setting to 0 will always attempt image subtraction."
    )
    doScaleVariance = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Scale variance of the image difference?"
    )
    scaleVariance = lsst.pex.config.ConfigurableField(
        target=ScaleVarianceTask,
        doc="Subtask to rescale the variance of the template to the statistically expected level."
    )
    doSubtractBackground = lsst.pex.config.Field(
        doc="Subtract the background fit when solving the kernel?",
        dtype=bool,
        default=True,
    )
    doApplyFinalizedPsf = lsst.pex.config.Field(
        doc="Replace science Exposure's psf and aperture correction map"
        " with those in finalizedPsfApCorrCatalog.",
        dtype=bool,
        default=False,
    )
    detectionThreshold = lsst.pex.config.Field(
        dtype=float,
        default=10,
        doc="Minimum signal to noise ration of detected sources "
        "to use for calculating the PSF matching kernel."
    )
    badSourceFlags = lsst.pex.config.ListField(
        dtype=str,
        doc="Flags that, if set, the associated source should not "
        "be used to determine the PSF matching kernel.",
        default=("sky_source", "slot_Centroid_flag",
                 "slot_ApFlux_flag", "slot_PsfFlux_flag", ),
    )
    badMaskPlanes = lsst.pex.config.ListField(
        dtype=str,
        default=("NO_DATA", "BAD", "SAT", "EDGE"),
        doc="Mask planes to exclude when selecting sources for PSF matching."
    )

    def setDefaults(self):
        self.makeKernel.kernel.name = "AL"
        self.makeKernel.kernel.active.fitForBackground = self.doSubtractBackground
        self.makeKernel.kernel.active.spatialKernelOrder = 1
        self.makeKernel.kernel.active.spatialBgOrder = 2


class AlardLuptonSubtractConfig(AlardLuptonSubtractBaseConfig, lsst.pipe.base.PipelineTaskConfig,
                                pipelineConnections=AlardLuptonSubtractConnections):
    mode = lsst.pex.config.ChoiceField(
        dtype=str,
        default="convolveTemplate",
        allowed={"auto": "Choose which image to convolve at runtime.",
                 "convolveScience": "Only convolve the science image.",
                 "convolveTemplate": "Only convolve the template image."},
        doc="Choose which image to convolve at runtime, or require that a specific image is convolved."
    )


class AlardLuptonSubtractTask(lsst.pipe.base.PipelineTask):
    """Compute the image difference of a science and template image using
    the Alard & Lupton (1998) algorithm.
    """
    ConfigClass = AlardLuptonSubtractConfig
    _DefaultName = "alardLuptonSubtract"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("decorrelate")
        self.makeSubtask("makeKernel")
        if self.config.doScaleVariance:
            self.makeSubtask("scaleVariance")

        self.convolutionControl = lsst.afw.math.ConvolutionControl()
        # Normalization is an extra, unnecessary, calculation and will result
        #  in mis-subtraction of the images if there are calibration errors.
        self.convolutionControl.setDoNormalize(False)
        self.convolutionControl.setDoCopyEdge(True)

    def _applyExternalCalibrations(self, exposure, finalizedPsfApCorrCatalog):
        """Replace calibrations (psf, and ApCorrMap) on this exposure with external ones.".

        Parameters
        ----------
        exposure : `lsst.afw.image.exposure.Exposure`
            Input exposure to adjust calibrations.
        finalizedPsfApCorrCatalog : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with finalized psf models and aperture correction
            maps to be applied if config.doApplyFinalizedPsf=True.  Catalog uses
            the detector id for the catalog id, sorted on id for fast lookup.

        Returns
        -------
        exposure : `lsst.afw.image.exposure.Exposure`
            Exposure with adjusted calibrations.
        """
        detectorId = exposure.info.getDetector().getId()

        row = finalizedPsfApCorrCatalog.find(detectorId)
        if row is None:
            self.log.warning("Detector id %s not found in finalizedPsfApCorrCatalog; "
                             "Using original psf.", detectorId)
        else:
            psf = row.getPsf()
            apCorrMap = row.getApCorrMap()
            if psf is None:
                self.log.warning("Detector id %s has None for psf in "
                                 "finalizedPsfApCorrCatalog; Using original psf and aperture correction.",
                                 detectorId)
            elif apCorrMap is None:
                self.log.warning("Detector id %s has None for apCorrMap in "
                                 "finalizedPsfApCorrCatalog; Using original psf and aperture correction.",
                                 detectorId)
            else:
                exposure.setPsf(psf)
                exposure.info.setApCorrMap(apCorrMap)

        return exposure

    @timeMethod
    def run(self, template, science, sources, finalizedPsfApCorrCatalog=None):
        """PSF match, subtract, and decorrelate two images.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            Template exposure, warped to match the science exposure.
        science : `lsst.afw.image.ExposureF`
            Science exposure to subtract from the template.
        sources : `lsst.afw.table.SourceCatalog`
            Identified sources on the science exposure. This catalog is used to
            select sources in order to perform the AL PSF matching on stamp
            images around them.
        finalizedPsfApCorrCatalog : `lsst.afw.table.ExposureCatalog`, optional
            Exposure catalog with finalized psf models and aperture correction
            maps to be applied if config.doApplyFinalizedPsf=True.  Catalog uses
            the detector id for the catalog id, sorted on id for fast lookup.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            ``difference`` : `lsst.afw.image.ExposureF`
                Result of subtracting template and science.
            ``matchedTemplate`` : `lsst.afw.image.ExposureF`
                Warped and PSF-matched template exposure.
            ``backgroundModel`` : `lsst.afw.math.Function2D`
                Background model that was fit while solving for the PSF-matching kernel
            ``psfMatchingKernel`` : `lsst.afw.math.Kernel`
                Kernel used to PSF-match the convolved image.

        Raises
        ------
        RuntimeError
            If an unsupported convolution mode is supplied.
        RuntimeError
            If there are too few sources to calculate the PSF matching kernel.
        lsst.pipe.base.NoWorkFound
            Raised if fraction of good pixels, defined as not having NO_DATA
            set, is less then the configured requiredTemplateFraction
        """
        self._prepareInputs(template, science,
                            finalizedPsfApCorrCatalog=finalizedPsfApCorrCatalog)

        # In the event that getPsfFwhm fails, evaluate the PSF on a grid.
        fwhmExposureBuffer = self.config.makeKernel.fwhmExposureBuffer
        fwhmExposureGrid = self.config.makeKernel.fwhmExposureGrid

        # Calling getPsfFwhm on template.psf fails on some rare occasions when
        # the template has no input exposures at the average position of the
        # stars. So we try getPsfFwhm first on template, and if that fails we
        # evaluate the PSF on a grid specified by fwhmExposure* fields.
        # To keep consistent definitions for PSF size on the template and
        # science images, we use the same method for both.
        try:
            templatePsfSize = getPsfFwhm(template.psf)
            sciencePsfSize = getPsfFwhm(science.psf)
        except InvalidParameterError:
            self.log.info("Unable to evaluate PSF at the average position. "
                          "Evaluting PSF on a grid of points."
                          )
            templatePsfSize = evaluateMeanPsfFwhm(template,
                                                  fwhmExposureBuffer=fwhmExposureBuffer,
                                                  fwhmExposureGrid=fwhmExposureGrid
                                                  )
            sciencePsfSize = evaluateMeanPsfFwhm(science,
                                                 fwhmExposureBuffer=fwhmExposureBuffer,
                                                 fwhmExposureGrid=fwhmExposureGrid
                                                 )
        self.log.info("Science PSF FWHM: %f pixels", sciencePsfSize)
        self.log.info("Template PSF FWHM: %f pixels", templatePsfSize)
        selectSources = self._sourceSelector(sources, science.mask)

        if self.config.mode == "auto":
            convolveTemplate = _shapeTest(template,
                                          science,
                                          fwhmExposureBuffer=fwhmExposureBuffer,
                                          fwhmExposureGrid=fwhmExposureGrid)
            if convolveTemplate:
                if sciencePsfSize < templatePsfSize:
                    self.log.info("Average template PSF size is greater, "
                                  "but science PSF greater in one dimension: convolving template image.")
                else:
                    self.log.info("Science PSF size is greater: convolving template image.")
            else:
                self.log.info("Template PSF size is greater: convolving science image.")
        elif self.config.mode == "convolveTemplate":
            self.log.info("`convolveTemplate` is set: convolving template image.")
            convolveTemplate = True
        elif self.config.mode == "convolveScience":
            self.log.info("`convolveScience` is set: convolving science image.")
            convolveTemplate = False
        else:
            raise RuntimeError("Cannot handle AlardLuptonSubtract mode: %s", self.config.mode)

        if convolveTemplate:
            subtractResults = self.runConvolveTemplate(template, science, selectSources)
        else:
            subtractResults = self.runConvolveScience(template, science, selectSources)

        return subtractResults

    def runConvolveTemplate(self, template, science, selectSources):
        """Convolve the template image with a PSF-matching kernel and subtract
        from the science image.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            Template exposure, warped to match the science exposure.
        science : `lsst.afw.image.ExposureF`
            Science exposure to subtract from the template.
        selectSources : `lsst.afw.table.SourceCatalog`
            Identified sources on the science exposure. This catalog is used to
            select sources in order to perform the AL PSF matching on stamp
            images around them.

        Returns
        -------
        results : `lsst.pipe.base.Struct`

            ``difference`` : `lsst.afw.image.ExposureF`
                Result of subtracting template and science.
            ``matchedTemplate`` : `lsst.afw.image.ExposureF`
                Warped and PSF-matched template exposure.
            ``backgroundModel`` : `lsst.afw.math.Function2D`
                Background model that was fit while solving for the PSF-matching kernel
            ``psfMatchingKernel`` : `lsst.afw.math.Kernel`
                Kernel used to PSF-match the template to the science image.
        """
        kernelSources = self.makeKernel.selectKernelSources(template, science,
                                                            candidateList=selectSources,
                                                            preconvolved=False)
        kernelResult = self.makeKernel.run(template, science, kernelSources,
                                           preconvolved=False)

        matchedTemplate = self._convolveExposure(template, kernelResult.psfMatchingKernel,
                                                 self.convolutionControl,
                                                 bbox=science.getBBox(),
                                                 psf=science.psf,
                                                 photoCalib=science.photoCalib)

        difference = _subtractImages(science, matchedTemplate,
                                     backgroundModel=(kernelResult.backgroundModel
                                                      if self.config.doSubtractBackground else None))
        correctedExposure = self.finalize(template, science, difference,
                                          kernelResult.psfMatchingKernel,
                                          templateMatched=True)

        return lsst.pipe.base.Struct(difference=correctedExposure,
                                     matchedTemplate=matchedTemplate,
                                     matchedScience=science,
                                     backgroundModel=kernelResult.backgroundModel,
                                     psfMatchingKernel=kernelResult.psfMatchingKernel)

    def runConvolveScience(self, template, science, selectSources):
        """Convolve the science image with a PSF-matching kernel and subtract the template image.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            Template exposure, warped to match the science exposure.
        science : `lsst.afw.image.ExposureF`
            Science exposure to subtract from the template.
        selectSources : `lsst.afw.table.SourceCatalog`
            Identified sources on the science exposure. This catalog is used to
            select sources in order to perform the AL PSF matching on stamp
            images around them.

        Returns
        -------
        results : `lsst.pipe.base.Struct`

            ``difference`` : `lsst.afw.image.ExposureF`
                Result of subtracting template and science.
            ``matchedTemplate`` : `lsst.afw.image.ExposureF`
                Warped template exposure. Note that in this case, the template
                is not PSF-matched to the science image.
            ``backgroundModel`` : `lsst.afw.math.Function2D`
                Background model that was fit while solving for the PSF-matching kernel
            ``psfMatchingKernel`` : `lsst.afw.math.Kernel`
               Kernel used to PSF-match the science image to the template.
        """
        bbox = science.getBBox()
        kernelSources = self.makeKernel.selectKernelSources(science, template,
                                                            candidateList=selectSources,
                                                            preconvolved=False)
        kernelResult = self.makeKernel.run(science, template, kernelSources,
                                           preconvolved=False)
        modelParams = kernelResult.backgroundModel.getParameters()
        # We must invert the background model if the matching kernel is solved for the science image.
        kernelResult.backgroundModel.setParameters([-p for p in modelParams])

        kernelImage = lsst.afw.image.ImageD(kernelResult.psfMatchingKernel.getDimensions())
        norm = kernelResult.psfMatchingKernel.computeImage(kernelImage, doNormalize=False)

        matchedScience = self._convolveExposure(science, kernelResult.psfMatchingKernel,
                                                self.convolutionControl,
                                                psf=template.psf)

        # Place back on native photometric scale
        matchedScience.maskedImage /= norm
        matchedTemplate = template.clone()[bbox]
        matchedTemplate.maskedImage /= norm
        matchedTemplate.setPhotoCalib(science.photoCalib)

        difference = _subtractImages(matchedScience, matchedTemplate,
                                     backgroundModel=(kernelResult.backgroundModel
                                                      if self.config.doSubtractBackground else None))

        correctedExposure = self.finalize(template, science, difference,
                                          kernelResult.psfMatchingKernel,
                                          templateMatched=False)

        return lsst.pipe.base.Struct(difference=correctedExposure,
                                     matchedTemplate=matchedTemplate,
                                     matchedScience=matchedScience,
                                     backgroundModel=kernelResult.backgroundModel,
                                     psfMatchingKernel=kernelResult.psfMatchingKernel,)

    def finalize(self, template, science, difference, kernel,
                 templateMatched=True,
                 preConvMode=False,
                 preConvKernel=None,
                 spatiallyVarying=False):
        """Decorrelate the difference image to undo the noise correlations
        caused by convolution.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            Template exposure, warped to match the science exposure.
        science : `lsst.afw.image.ExposureF`
            Science exposure to subtract from the template.
        difference : `lsst.afw.image.ExposureF`
            Result of subtracting template and science.
        kernel : `lsst.afw.math.Kernel`
            An (optionally spatially-varying) PSF matching kernel
        templateMatched : `bool`, optional
            Was the template PSF-matched to the science image?
        preConvMode : `bool`, optional
            Was the science image preconvolved with its own PSF
            before PSF matching the template?
        preConvKernel : `lsst.afw.detection.Psf`, optional
            If not `None`, then the science image was pre-convolved with
            (the reflection of) this kernel. Must be normalized to sum to 1.
        spatiallyVarying : `bool`, optional
            Compute the decorrelation kernel spatially varying across the image?

        Returns
        -------
        correctedExposure : `lsst.afw.image.ExposureF`
            The decorrelated image difference.
        """
        # Erase existing detection mask planes.
        #  We don't want the detection mask from the science image
        mask = difference.mask
        mask &= ~(mask.getPlaneBitMask("DETECTED") | mask.getPlaneBitMask("DETECTED_NEGATIVE"))

        if self.config.doDecorrelation:
            self.log.info("Decorrelating image difference.")
            correctedExposure = self.decorrelate.run(science, template[science.getBBox()], difference, kernel,
                                                     templateMatched=templateMatched,
                                                     preConvMode=preConvMode,
                                                     preConvKernel=preConvKernel,
                                                     spatiallyVarying=spatiallyVarying).correctedExposure
        else:
            self.log.info("NOT decorrelating image difference.")
            correctedExposure = difference
        return correctedExposure

    @staticmethod
    def _validateExposures(template, science):
        """Check that the WCS of the two Exposures match, and the template bbox
        contains the science bbox.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            Template exposure, warped to match the science exposure.
        science : `lsst.afw.image.ExposureF`
            Science exposure to subtract from the template.

        Raises
        ------
        AssertionError
            Raised if the WCS of the template is not equal to the science WCS,
            or if the science image is not fully contained in the template
            bounding box.
        """
        assert template.wcs == science.wcs,\
            "Template and science exposure WCS are not identical."
        templateBBox = template.getBBox()
        scienceBBox = science.getBBox()

        assert templateBBox.contains(scienceBBox),\
            "Template bbox does not contain all of the science image."

    @staticmethod
    def _convolveExposure(exposure, kernel, convolutionControl,
                          bbox=None,
                          psf=None,
                          photoCalib=None):
        """Convolve an exposure with the given kernel.

        Parameters
        ----------
        exposure : `lsst.afw.Exposure`
            exposure to convolve.
        kernel : `lsst.afw.math.LinearCombinationKernel`
            PSF matching kernel computed in the ``makeKernel`` subtask.
        convolutionControl : `lsst.afw.math.ConvolutionControl`
            Configuration for convolve algorithm.
        bbox : `lsst.geom.Box2I`, optional
            Bounding box to trim the convolved exposure to.
        psf : `lsst.afw.detection.Psf`, optional
            Point spread function (PSF) to set for the convolved exposure.
        photoCalib : `lsst.afw.image.PhotoCalib`, optional
            Photometric calibration of the convolved exposure.

        Returns
        -------
        convolvedExp : `lsst.afw.Exposure`
            The convolved image.
        """
        convolvedExposure = exposure.clone()
        if psf is not None:
            convolvedExposure.setPsf(psf)
        if photoCalib is not None:
            convolvedExposure.setPhotoCalib(photoCalib)
        convolvedImage = lsst.afw.image.MaskedImageF(exposure.getBBox())
        lsst.afw.math.convolve(convolvedImage, exposure.maskedImage, kernel, convolutionControl)
        convolvedExposure.setMaskedImage(convolvedImage)
        if bbox is None:
            return convolvedExposure
        else:
            return convolvedExposure[bbox]

    def _sourceSelector(self, sources, mask):
        """Select sources from a catalog that meet the selection criteria.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog`
            Input source catalog to select sources from.
        mask : `lsst.afw.image.Mask`
            The image mask plane to use to reject sources
            based on their location on the ccd.

        Returns
        -------
        selectSources : `lsst.afw.table.SourceCatalog`
            The input source catalog, with flagged and low signal-to-noise
            sources removed.

        Raises
        ------
        RuntimeError
            If there are too few sources to compute the PSF matching kernel
            remaining after source selection.
        """
        flags = np.ones(len(sources), dtype=bool)
        for flag in self.config.badSourceFlags:
            try:
                flags *= ~sources[flag]
            except Exception as e:
                self.log.warning("Could not apply source flag: %s", e)
        sToNFlag = (sources.getPsfInstFlux()/sources.getPsfInstFluxErr()) > self.config.detectionThreshold
        flags *= sToNFlag
        flags *= self._checkMask(mask, sources, self.config.badMaskPlanes)
        selectSources = sources[flags]
        self.log.info("%i/%i=%.1f%% of sources selected for PSF matching from the input catalog",
                      len(selectSources), len(sources), 100*len(selectSources)/len(sources))
        if len(selectSources) < self.config.makeKernel.nStarPerCell:
            self.log.error("Too few sources to calculate the PSF matching kernel: "
                           "%i selected but %i needed for the calculation.",
                           len(selectSources), self.config.makeKernel.nStarPerCell)
            raise RuntimeError("Cannot compute PSF matching kernel: too few sources selected.")

        return selectSources.copy(deep=True)

    @staticmethod
    def _checkMask(mask, sources, badMaskPlanes):
        """Exclude sources that are located on masked pixels.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            The image mask plane to use to reject sources
            based on the location of their centroid on the ccd.
        sources : `lsst.afw.table.SourceCatalog`
            The source catalog to evaluate.
        badMaskPlanes : `list` of `str`
            List of the names of the mask planes to exclude.

        Returns
        -------
        flags : `numpy.ndarray` of `bool`
            Array indicating whether each source in the catalog should be
            kept (True) or rejected (False) based on the value of the
            mask plane at its location.
        """
        badPixelMask = lsst.afw.image.Mask.getPlaneBitMask(badMaskPlanes)
        xv = np.rint(sources.getX() - mask.getX0())
        yv = np.rint(sources.getY() - mask.getY0())

        mv = mask.array[yv.astype(int), xv.astype(int)]
        flags = np.bitwise_and(mv, badPixelMask) == 0
        return flags

    def _prepareInputs(self, template, science,
                       finalizedPsfApCorrCatalog=None):
        """Perform preparatory calculations common to all Alard&Lupton Tasks.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            Template exposure, warped to match the science exposure.
            The variance plane of the template image is modified in place.
        science : `lsst.afw.image.ExposureF`
            Science exposure to subtract from the template.
            The variance plane of the science image is modified in place.
        finalizedPsfApCorrCatalog : `lsst.afw.table.ExposureCatalog`, optional
            Exposure catalog with finalized psf models and aperture correction
            maps to be applied if config.doApplyFinalizedPsf=True.  Catalog uses
            the detector id for the catalog id, sorted on id for fast lookup.
        """
        self._validateExposures(template, science)
        if self.config.doApplyFinalizedPsf:
            self._applyExternalCalibrations(science,
                                            finalizedPsfApCorrCatalog=finalizedPsfApCorrCatalog)
        checkTemplateIsSufficient(template, self.log,
                                  requiredTemplateFraction=self.config.requiredTemplateFraction)

        if self.config.doScaleVariance:
            # Scale the variance of the template and science images before
            # convolution, subtraction, or decorrelation so that they have the
            # correct ratio.
            templateVarFactor = self.scaleVariance.run(template.maskedImage)
            sciVarFactor = self.scaleVariance.run(science.maskedImage)
            self.log.info("Template variance scaling factor: %.2f", templateVarFactor)
            self.metadata.add("scaleTemplateVarianceFactor", templateVarFactor)
            self.log.info("Science variance scaling factor: %.2f", sciVarFactor)
            self.metadata.add("scaleScienceVarianceFactor", sciVarFactor)


class AlardLuptonPreconvolveSubtractConnections(SubtractInputConnections,
                                                SubtractScoreOutputConnections):
    pass


class AlardLuptonPreconvolveSubtractConfig(AlardLuptonSubtractBaseConfig, lsst.pipe.base.PipelineTaskConfig,
                                           pipelineConnections=AlardLuptonPreconvolveSubtractConnections):
    pass


class AlardLuptonPreconvolveSubtractTask(AlardLuptonSubtractTask):
    """Subtract a template from a science image, convolving the science image
    before computing the kernel, and also convolving the template before
    subtraction.
    """
    ConfigClass = AlardLuptonPreconvolveSubtractConfig
    _DefaultName = "alardLuptonPreconvolveSubtract"

    def run(self, template, science, sources, finalizedPsfApCorrCatalog=None):
        """Preconvolve the science image with its own PSF,
        convolve the template image with a PSF-matching kernel and subtract
        from the preconvolved science image.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            The template image, which has previously been warped to
            the science image. The template bbox will be padded by a few pixels
            compared to the science bbox.
        science : `lsst.afw.image.ExposureF`
            The science exposure.
        sources : `lsst.afw.table.SourceCatalog`
            Identified sources on the science exposure. This catalog is used to
            select sources in order to perform the AL PSF matching on stamp
            images around them.
        finalizedPsfApCorrCatalog : `lsst.afw.table.ExposureCatalog`, optional
            Exposure catalog with finalized psf models and aperture correction
            maps to be applied if config.doApplyFinalizedPsf=True.  Catalog uses
            the detector id for the catalog id, sorted on id for fast lookup.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            ``scoreExposure`` : `lsst.afw.image.ExposureF`
                Result of subtracting the convolved template and science images.
                Attached PSF is that of the original science image.
            ``matchedTemplate`` : `lsst.afw.image.ExposureF`
                Warped and PSF-matched template exposure.
                Attached PSF is that of the original science image.
            ``matchedScience`` : `lsst.afw.image.ExposureF`
                The science exposure after convolving with its own PSF.
                Attached PSF is that of the original science image.
            ``backgroundModel`` : `lsst.afw.math.Function2D`
                Background model that was fit while solving for the PSF-matching kernel
            ``psfMatchingKernel`` : `lsst.afw.math.Kernel`
                Final kernel used to PSF-match the template to the science image.
        """
        self._prepareInputs(template, science,
                            finalizedPsfApCorrCatalog=finalizedPsfApCorrCatalog)

        # TODO: DM-37212 we need to mirror the kernel in order to get correct cross correlation
        scienceKernel = science.psf.getKernel()
        matchedScience = self._convolveExposure(science, scienceKernel, self.convolutionControl)
        selectSources = self._sourceSelector(sources, matchedScience.mask)

        subtractResults = self.runPreconvolve(template, science, matchedScience, selectSources, scienceKernel)

        return subtractResults

    def runPreconvolve(self, template, science, matchedScience, selectSources, preConvKernel):
        """Convolve the science image with its own PSF, then convolve the
        template with a matching kernel and subtract to form the Score exposure.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            Template exposure, warped to match the science exposure.
        science : `lsst.afw.image.ExposureF`
            Science exposure to subtract from the template.
        matchedScience : `lsst.afw.image.ExposureF`
            The science exposure, convolved with the reflection of its own PSF.
        selectSources : `lsst.afw.table.SourceCatalog`
            Identified sources on the science exposure. This catalog is used to
            select sources in order to perform the AL PSF matching on stamp
            images around them.
        preConvKernel : `lsst.afw.math.Kernel`
            The reflection of the kernel that was used to preconvolve
            the `science` exposure.
            Must be normalized to sum to 1.

        Returns
        -------
        results : `lsst.pipe.base.Struct`

            ``scoreExposure`` : `lsst.afw.image.ExposureF`
                Result of subtracting the convolved template and science images.
                Attached PSF is that of the original science image.
            ``matchedTemplate`` : `lsst.afw.image.ExposureF`
                Warped and PSF-matched template exposure.
                Attached PSF is that of the original science image.
            ``matchedScience`` : `lsst.afw.image.ExposureF`
                The science exposure after convolving with its own PSF.
                Attached PSF is that of the original science image.
            ``backgroundModel`` : `lsst.afw.math.Function2D`
                Background model that was fit while solving for the PSF-matching kernel
            ``psfMatchingKernel`` : `lsst.afw.math.Kernel`
                Final kernel used to PSF-match the template to the science image.
        """
        bbox = science.getBBox()
        innerBBox = preConvKernel.shrinkBBox(bbox)

        kernelSources = self.makeKernel.selectKernelSources(template[innerBBox], science[innerBBox],
                                                            candidateList=selectSources,
                                                            preconvolved=True)
        kernelResult = self.makeKernel.run(template[innerBBox], matchedScience[innerBBox], kernelSources,
                                           preconvolved=True)

        matchedTemplate = self._convolveExposure(template, kernelResult.psfMatchingKernel,
                                                 self.convolutionControl,
                                                 bbox=bbox,
                                                 psf=science.psf,
                                                 photoCalib=science.photoCalib)
        score = _subtractImages(matchedScience, matchedTemplate,
                                backgroundModel=(kernelResult.backgroundModel
                                                 if self.config.doSubtractBackground else None))
        correctedScore = self.finalize(template[bbox], science, score,
                                       kernelResult.psfMatchingKernel,
                                       templateMatched=True, preConvMode=True,
                                       preConvKernel=preConvKernel)

        return lsst.pipe.base.Struct(scoreExposure=correctedScore,
                                     matchedTemplate=matchedTemplate,
                                     matchedScience=matchedScience,
                                     backgroundModel=kernelResult.backgroundModel,
                                     psfMatchingKernel=kernelResult.psfMatchingKernel)


def checkTemplateIsSufficient(templateExposure, logger, requiredTemplateFraction=0.):
    """Raise NoWorkFound if template coverage < requiredTemplateFraction

    Parameters
    ----------
    templateExposure : `lsst.afw.image.ExposureF`
        The template exposure to check
    logger : `lsst.log.Log`
        Logger for printing output.
    requiredTemplateFraction : `float`, optional
        Fraction of pixels of the science image required to have coverage
        in the template.

    Raises
    ------
    lsst.pipe.base.NoWorkFound
        Raised if fraction of good pixels, defined as not having NO_DATA
        set, is less then the configured requiredTemplateFraction
    """
    # Count the number of pixels with the NO_DATA mask bit set
    # counting NaN pixels is insufficient because pixels without data are often intepolated over)
    pixNoData = np.count_nonzero(templateExposure.mask.array
                                 & templateExposure.mask.getPlaneBitMask('NO_DATA'))
    pixGood = templateExposure.getBBox().getArea() - pixNoData
    logger.info("template has %d good pixels (%.1f%%)", pixGood,
                100*pixGood/templateExposure.getBBox().getArea())

    if pixGood/templateExposure.getBBox().getArea() < requiredTemplateFraction:
        message = ("Insufficient Template Coverage. (%.1f%% < %.1f%%) Not attempting subtraction. "
                   "To force subtraction, set config requiredTemplateFraction=0." % (
                       100*pixGood/templateExposure.getBBox().getArea(),
                       100*requiredTemplateFraction))
        raise lsst.pipe.base.NoWorkFound(message)


def _subtractImages(science, template, backgroundModel=None):
    """Subtract template from science, propagating relevant metadata.

    Parameters
    ----------
    science : `lsst.afw.Exposure`
        The input science image.
    template : `lsst.afw.Exposure`
        The template to subtract from the science image.
    backgroundModel : `lsst.afw.MaskedImage`, optional
        Differential background model

    Returns
    -------
    difference : `lsst.afw.Exposure`
        The subtracted image.
    """
    difference = science.clone()
    if backgroundModel is not None:
        difference.maskedImage -= backgroundModel
    difference.maskedImage -= template.maskedImage
    return difference


def _shapeTest(exp1, exp2, fwhmExposureBuffer, fwhmExposureGrid):
    """Determine that the PSF of ``exp1`` is not wider than that of ``exp2``.

    Parameters
    ----------
    exp1 : `~lsst.afw.image.Exposure`
        Exposure with the reference point spread function (PSF) to evaluate.
    exp2 : `~lsst.afw.image.Exposure`
        Exposure with a candidate point spread function (PSF) to evaluate.
    fwhmExposureBuffer : `float`
        Fractional buffer margin to be left out of all sides of the image
        during the construction of the grid to compute mean PSF FWHM in an
        exposure, if the PSF is not available at its average position.
    fwhmExposureGrid : `int`
        Grid size to compute the mean FWHM in an exposure, if the PSF is not
        available at its average position.
    Returns
    -------
    result : `bool`
        True if ``exp1`` has a PSF that is not wider than that of ``exp2`` in
        either dimension.
    """
    try:
        shape1 = getPsfFwhm(exp1.psf, average=False)
        shape2 = getPsfFwhm(exp2.psf, average=False)
    except InvalidParameterError:
        shape1 = evaluateMeanPsfFwhm(exp1,
                                     fwhmExposureBuffer=fwhmExposureBuffer,
                                     fwhmExposureGrid=fwhmExposureGrid
                                     )
        shape2 = evaluateMeanPsfFwhm(exp2,
                                     fwhmExposureBuffer=fwhmExposureBuffer,
                                     fwhmExposureGrid=fwhmExposureGrid
                                     )
        return shape1 <= shape2

    # Results from getPsfFwhm is a tuple of two values, one for each dimension.
    xTest = shape1[0] <= shape2[0]
    yTest = shape1[1] <= shape2[1]
    return xTest | yTest
