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

import lsst.afw.detection as afwDetection
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.geom
from lsst.ip.diffim.utils import evaluateMaskFraction
from lsst.meas.algorithms import SkyObjectsTask, SourceDetectionTask, SetPrimaryFlagsTask, MaskStreaksTask
from lsst.meas.base import ForcedMeasurementTask, ApplyApCorrTask, DetectorVisitIdGeneratorConfig
import lsst.meas.deblender
import lsst.meas.extensions.trailedSources  # noqa: F401
import lsst.meas.extensions.shapeHSM
import lsst.pex.config as pexConfig
from lsst.pex.exceptions import InvalidParameterError
import lsst.pipe.base as pipeBase
import lsst.utils
from lsst.utils.timer import timeMethod

from . import DipoleFitTask

__all__ = ["DetectAndMeasureConfig", "DetectAndMeasureTask",
           "DetectAndMeasureScoreConfig", "DetectAndMeasureScoreTask"]


class DetectAndMeasureConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=("instrument", "visit", "detector"),
                                  defaultTemplates={"coaddName": "deep",
                                                    "warpTypeSuffix": "",
                                                    "fakesType": ""}):
    science = pipeBase.connectionTypes.Input(
        doc="Input science exposure.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}calexp"
    )
    matchedTemplate = pipeBase.connectionTypes.Input(
        doc="Warped and PSF-matched template used to create the difference image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_matchedExp",
    )
    difference = pipeBase.connectionTypes.Input(
        doc="Result of subtracting template from science.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_differenceTempExp",
    )
    outputSchema = pipeBase.connectionTypes.InitOutput(
        doc="Schema (as an example catalog) for output DIASource catalog.",
        storageClass="SourceCatalog",
        name="{fakesType}{coaddName}Diff_diaSrc_schema",
    )
    diaSources = pipeBase.connectionTypes.Output(
        doc="Detected diaSources on the difference image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        name="{fakesType}{coaddName}Diff_diaSrc",
    )
    subtractedMeasuredExposure = pipeBase.connectionTypes.Output(
        doc="Difference image with detection mask plane filled in.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_differenceExp",
    )
    maskedStreaks = pipeBase.connectionTypes.Output(
        doc='Catalog of streak fit parameters for the difference image.',
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "visit", "detector"),
        name="{fakesType}{coaddName}Diff_streaks",
    )

    def __init__(self, *, config):
        super().__init__(config=config)
        if not (self.config.writeStreakInfo and self.config.doMaskStreaks):
            self.outputs.remove("maskedStreaks")


class DetectAndMeasureConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=DetectAndMeasureConnections):
    """Config for DetectAndMeasureTask
    """
    doMerge = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Merge positive and negative diaSources with grow radius "
            "set by growFootprint"
    )
    doForcedMeasurement = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Force photometer diaSource locations on PVI?")
    doAddMetrics = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Add columns to the source table to hold analysis metrics?"
    )
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Final source detection for diaSource measurement",
    )
    streakDetection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Separate source detection used only for streak masking",
    )
    deblend = pexConfig.ConfigurableField(
        target=lsst.meas.deblender.SourceDeblendTask,
        doc="Task to split blended sources into their components."
    )
    measurement = pexConfig.ConfigurableField(
        target=DipoleFitTask,
        doc="Task to measure sources on the difference image.",
    )
    doApCorr = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Run subtask to apply aperture corrections"
    )
    applyApCorr = lsst.pex.config.ConfigurableField(
        target=ApplyApCorrTask,
        doc="Task to apply aperture corrections"
    )
    forcedMeasurement = pexConfig.ConfigurableField(
        target=ForcedMeasurementTask,
        doc="Task to force photometer science image at diaSource locations.",
    )
    growFootprint = pexConfig.Field(
        dtype=int,
        default=2,
        doc="Grow positive and negative footprints by this many pixels before merging"
    )
    diaSourceMatchRadius = pexConfig.Field(
        dtype=float,
        default=0.5,
        doc="Match radius (in arcseconds) for DiaSource to Source association"
    )
    doSkySources = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Generate sky sources?",
    )
    skySources = pexConfig.ConfigurableField(
        target=SkyObjectsTask,
        doc="Generate sky sources",
    )
    doMaskStreaks = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Turn on streak masking",
    )
    maskStreaks = pexConfig.ConfigurableField(
        target=MaskStreaksTask,
        doc="Subtask for masking streaks. Only used if doMaskStreaks is True. "
            "Adds a mask plane to an exposure, with the mask plane name set by streakMaskName.",
    )
    streakBinFactor = pexConfig.Field(
        dtype=int,
        default=4,
        doc="Bin scale factor to use when rerunning detection for masking streaks. "
            "Only used if doMaskStreaks is True.",
    )
    writeStreakInfo = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Record the parameters of any detected streaks. For LSST, this should be turned off except for "
            "development work."
    )
    setPrimaryFlags = pexConfig.ConfigurableField(
        target=SetPrimaryFlagsTask,
        doc="Task to add isPrimary and deblending-related flags to the catalog."
    )
    badSourceFlags = lsst.pex.config.ListField(
        dtype=str,
        doc="Sources with any of these flags set are removed before writing the output catalog.",
        default=("base_PixelFlags_flag_offimage",
                 "base_PixelFlags_flag_interpolatedCenterAll",
                 "base_PixelFlags_flag_badCenterAll",
                 "base_PixelFlags_flag_edgeCenterAll",
                 "base_PixelFlags_flag_saturatedCenterAll",
                 ),
    )
    clearMaskPlanes = lsst.pex.config.ListField(
        dtype=str,
        doc="Mask planes to clear before running detection.",
        default=("DETECTED", "DETECTED_NEGATIVE", "NOT_DEBLENDED", "STREAK"),
    )
    idGenerator = DetectorVisitIdGeneratorConfig.make_field()

    def setDefaults(self):
        # DiaSource Detection
        self.detection.thresholdPolarity = "both"
        self.detection.thresholdValue = 5.0
        self.detection.reEstimateBackground = False
        self.detection.thresholdType = "pixel_stdev"
        self.detection.excludeMaskPlanes = ["EDGE",
                                            "SAT",
                                            "BAD",
                                            "INTRP",
                                            "NO_DATA",
                                            ]

        # Copy configs for binned streak detection from the base detection task
        self.streakDetection.thresholdType = self.detection.thresholdType
        self.streakDetection.reEstimateBackground = self.detection.reEstimateBackground
        self.streakDetection.excludeMaskPlanes = self.detection.excludeMaskPlanes
        self.streakDetection.thresholdValue = self.detection.thresholdValue
        # Only detect positive streaks
        self.streakDetection.thresholdPolarity = "positive"
        # Do not grow detected mask for streaks
        self.streakDetection.nSigmaToGrow = 0
        # Set the streak mask along the entire fit line, not only where the
        # detected mask is set.
        self.maskStreaks.onlyMaskDetected = False
        # Restrict streak masking from growing too large
        self.maskStreaks.maxStreakWidth = 100
        # Restrict the number of iterations allowed for fitting streaks
        # When the fit is good it should solve quickly, and exit a bad fit quickly
        self.maskStreaks.maxFitIter = 10
        # Only mask to 2 sigma in width
        self.maskStreaks.nSigmaMask = 2
        # Threshold for including streaks after the Hough Transform.
        # A lower value will detect more features that are less linear.
        self.maskStreaks.absMinimumKernelHeight = 2

        self.measurement.plugins.names |= ["ext_trailedSources_Naive",
                                           "base_LocalPhotoCalib",
                                           "base_LocalWcs",
                                           "ext_shapeHSM_HsmSourceMoments",
                                           "ext_shapeHSM_HsmPsfMoments",
                                           "base_ClassificationSizeExtendedness",
                                           ]
        self.measurement.slots.psfShape = "ext_shapeHSM_HsmPsfMoments"
        self.measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"
        self.measurement.plugins["base_SdssCentroid"].maxDistToPeak = 5.0
        self.forcedMeasurement.plugins = ["base_TransformedCentroid", "base_PsfFlux"]
        self.forcedMeasurement.copyColumns = {
            "id": "objectId", "parent": "parentObjectId", "coord_ra": "coord_ra", "coord_dec": "coord_dec"}
        self.forcedMeasurement.slots.centroid = "base_TransformedCentroid"
        self.forcedMeasurement.slots.shape = None

        # Keep track of which footprints contain streaks
        self.measurement.plugins["base_PixelFlags"].masksFpAnywhere = [
            "STREAK", "INJECTED", "INJECTED_TEMPLATE"]
        self.measurement.plugins["base_PixelFlags"].masksFpCenter = [
            "STREAK", "INJECTED", "INJECTED_TEMPLATE"]
        self.skySources.avoidMask = ["DETECTED", "DETECTED_NEGATIVE", "BAD", "NO_DATA", "EDGE"]


class DetectAndMeasureTask(lsst.pipe.base.PipelineTask):
    """Detect and measure sources on a difference image.
    """
    ConfigClass = DetectAndMeasureConfig
    _DefaultName = "detectAndMeasure"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        # Add coordinate error fields:
        afwTable.CoordKey.addErrorFields(self.schema)

        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("detection", schema=self.schema)
        self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask("setPrimaryFlags", schema=self.schema, isSingleFrame=True)
        self.makeSubtask("measurement", schema=self.schema,
                         algMetadata=self.algMetadata)
        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.measurement.schema)
        if self.config.doForcedMeasurement:
            self.schema.addField(
                "ip_diffim_forced_PsfFlux_instFlux", "D",
                "Forced PSF flux measured on the direct image.",
                units="count")
            self.schema.addField(
                "ip_diffim_forced_PsfFlux_instFluxErr", "D",
                "Forced PSF flux error measured on the direct image.",
                units="count")
            self.schema.addField(
                "ip_diffim_forced_PsfFlux_area", "F",
                "Forced PSF flux effective area of PSF.",
                units="pixel")
            self.schema.addField(
                "ip_diffim_forced_PsfFlux_flag", "Flag",
                "Forced PSF flux general failure flag.")
            self.schema.addField(
                "ip_diffim_forced_PsfFlux_flag_noGoodPixels", "Flag",
                "Forced PSF flux not enough non-rejected pixels in data to attempt the fit.")
            self.schema.addField(
                "ip_diffim_forced_PsfFlux_flag_edge", "Flag",
                "Forced PSF flux object was too close to the edge of the image to use the full PSF model.")
            self.makeSubtask("forcedMeasurement", refSchema=self.schema)

        self.schema.addField("refMatchId", "L", "unique id of reference catalog match")
        self.schema.addField("srcMatchId", "L", "unique id of source match")
        if self.config.doSkySources:
            self.makeSubtask("skySources", schema=self.schema)
        if self.config.doMaskStreaks:
            self.makeSubtask("maskStreaks")
            self.makeSubtask("streakDetection")

        # Check that the schema and config are consistent
        for flag in self.config.badSourceFlags:
            if flag not in self.schema:
                raise pipeBase.InvalidQuantumError("Field %s not in schema" % flag)

        # initialize InitOutputs
        self.outputSchema = afwTable.SourceCatalog(self.schema)
        self.outputSchema.getTable().setMetadata(self.algMetadata)

    def runQuantum(self, butlerQC: pipeBase.QuantumContext,
                   inputRefs: pipeBase.InputQuantizedConnection,
                   outputRefs: pipeBase.OutputQuantizedConnection):
        inputs = butlerQC.get(inputRefs)
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        idFactory = idGenerator.make_table_id_factory()
        outputs = self.run(**inputs, idFactory=idFactory)
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(self, science, matchedTemplate, difference,
            idFactory=None):
        """Detect and measure sources on a difference image.

        The difference image will be convolved with a gaussian approximation of
        the PSF to form a maximum likelihood image for detection.
        Close positive and negative detections will optionally be merged into
        dipole diaSources.
        Sky sources, or forced detections in background regions, will optionally
        be added, and the configured measurement algorithm will be run on all
        detections.

        Parameters
        ----------
        science : `lsst.afw.image.ExposureF`
            Science exposure that the template was subtracted from.
        matchedTemplate : `lsst.afw.image.ExposureF`
            Warped and PSF-matched template that was used produce the
            difference image.
        difference : `lsst.afw.image.ExposureF`
            Result of subtracting template from the science image.
        idFactory : `lsst.afw.table.IdFactory`, optional
            Generator object used to assign ids to detected sources in the
            difference image. Ids from this generator are not set until after
            deblending and merging positive/negative peaks.

        Returns
        -------
        measurementResults : `lsst.pipe.base.Struct`

            ``subtractedMeasuredExposure`` : `lsst.afw.image.ExposureF`
                Subtracted exposure with detection mask applied.
            ``diaSources``  : `lsst.afw.table.SourceCatalog`
                The catalog of detected sources.
        """
        if idFactory is None:
            idFactory = lsst.meas.base.IdGenerator().make_table_id_factory()

        self._prepareInputs(difference)

        # Don't use the idFactory until after deblend+merge, so that we aren't
        # generating ids that just get thrown away (footprint merge doesn't
        # know about past ids).
        table = afwTable.SourceTable.make(self.schema)
        results = self.detection.run(
            table=table,
            exposure=difference,
            doSmooth=True,
        )

        sources, positives, negatives = self._deblend(difference,
                                                      results.positive,
                                                      results.negative)

        return self.processResults(science, matchedTemplate, difference, sources, idFactory,
                                   positiveFootprints=positives,
                                   negativeFootprints=negatives)

    def _prepareInputs(self, difference):
        """Ensure that we start with an empty detection and deblended mask.

        Parameters
        ----------
        difference : `lsst.afw.image.ExposureF`
            The difference image that will be used for detecting diaSources.
            The mask plane will be modified in place.

        Raises
        ------
        lsst.pipe.base.UpstreamFailureNoWorkFound
            If the PSF is not usable for measurement.
        """
        # Check that we have a valid PSF now before we do more work
        sigma = difference.psf.computeShape(difference.psf.getAveragePosition()).getDeterminantRadius()
        if np.isnan(sigma):
            raise pipeBase.UpstreamFailureNoWorkFound("Invalid PSF detected! PSF width evaluates to NaN.")
        # Ensure that we start with an empty detection and deblended mask.
        mask = difference.mask
        for mp in self.config.clearMaskPlanes:
            if mp not in mask.getMaskPlaneDict():
                mask.addMaskPlane(mp)
        mask &= ~mask.getPlaneBitMask(self.config.clearMaskPlanes)

    def processResults(self, science, matchedTemplate, difference, sources, idFactory,
                       positiveFootprints=None, negativeFootprints=None,):
        """Measure and process the results of source detection.

        Parameters
        ----------
        science : `lsst.afw.image.ExposureF`
            Science exposure that the template was subtracted from.
        matchedTemplate : `lsst.afw.image.ExposureF`
            Warped and PSF-matched template that was used produce the
            difference image.
        difference : `lsst.afw.image.ExposureF`
            Result of subtracting template from the science image.
        sources : `lsst.afw.table.SourceCatalog`
            Detected sources on the difference exposure.
        idFactory : `lsst.afw.table.IdFactory`
            Generator object used to assign ids to detected sources in the
            difference image.
        positiveFootprints : `lsst.afw.detection.FootprintSet`, optional
            Positive polarity footprints.
        negativeFootprints : `lsst.afw.detection.FootprintSet`, optional
            Negative polarity footprints.

        Returns
        -------
        measurementResults : `lsst.pipe.base.Struct`

            ``subtractedMeasuredExposure`` : `lsst.afw.image.ExposureF`
                Subtracted exposure with detection mask applied.
            ``diaSources``  : `lsst.afw.table.SourceCatalog`
                The catalog of detected sources.
        """
        self.metadata["nUnmergedDiaSources"] = len(sources)
        if self.config.doMerge:
            fpSet = positiveFootprints
            fpSet.merge(negativeFootprints, self.config.growFootprint,
                        self.config.growFootprint, False)
            initialDiaSources = afwTable.SourceCatalog(self.schema)
            fpSet.makeSources(initialDiaSources)
            self.log.info("Merging detections into %d sources", len(initialDiaSources))
        else:
            initialDiaSources = sources

        # Assign source ids at the end: deblend/merge mean that we don't keep
        # track of parents and children, we only care about the final ids.
        for source in initialDiaSources:
            source.setId(idFactory())
        # Ensure sources added after this get correct ids.
        initialDiaSources.getTable().setIdFactory(idFactory)
        initialDiaSources.setMetadata(self.algMetadata)

        self.metadata["nMergedDiaSources"] = len(initialDiaSources)

        if self.config.doMaskStreaks:
            streakInfo = self._runStreakMasking(difference)

        if self.config.doSkySources:
            self.addSkySources(initialDiaSources, difference.mask, difference.info.id)

        if not initialDiaSources.isContiguous():
            initialDiaSources = initialDiaSources.copy(deep=True)

        self.measureDiaSources(initialDiaSources, science, difference, matchedTemplate)
        diaSources = self._removeBadSources(initialDiaSources)

        if self.config.doForcedMeasurement:
            self.measureForcedSources(diaSources, science, difference.getWcs())

        self.calculateMetrics(difference)

        measurementResults = pipeBase.Struct(
            subtractedMeasuredExposure=difference,
            diaSources=diaSources,
        )
        if self.config.doMaskStreaks and self.config.writeStreakInfo:
            measurementResults.mergeItems(streakInfo, 'maskedStreaks')

        return measurementResults

    def _deblend(self, difference, positiveFootprints, negativeFootprints):
        """Deblend the positive and negative footprints and return a catalog
        containing just the children, and the deblended footprints.

        Parameters
        ----------
        difference : `lsst.afw.image.Exposure`
            Result of subtracting template from the science image.
        positiveFootprints, negativeFootprints : `lsst.afw.detection.FootprintSet`
            Positive and negative polarity footprints measured on
            ``difference`` to be deblended separately.

        Returns
        -------
        sources : `lsst.afw.table.SourceCatalog`
            Positive and negative deblended children.
        positives, negatives : `lsst.afw.detection.FootprintSet`
            Deblended positive and negative polarity footprints measured on
            ``difference``.
        """
        def makeFootprints(sources):
            footprints = afwDetection.FootprintSet(difference.getBBox())
            footprints.setFootprints([src.getFootprint() for src in sources])
            return footprints

        def deblend(footprints, negative=False):
            """Deblend a positive or negative footprint set,
            and return the deblended children.

            Parameters
            ----------
            footprints : `lsst.afw.detection.FootprintSet`
            negative : `bool`
                Set True if the footprints contain negative fluxes

            Returns
            -------
            sources : `lsst.afw.table.SourceCatalog`
            """
            sources = afwTable.SourceCatalog(self.schema)
            footprints.makeSources(sources)
            if negative:
                # Invert the image so the deblender can run on positive peaks
                difference_inverted = difference.clone()
                difference_inverted.image *= -1
                self.deblend.run(exposure=difference_inverted, sources=sources)
                children = sources[sources["parent"] != 0]
                # Set the heavy footprint pixel values back to reality
                for child in children:
                    footprint = child.getFootprint()
                    array = footprint.getImageArray()
                    array *= -1
            else:
                self.deblend.run(exposure=difference, sources=sources)
            self.setPrimaryFlags.run(sources)
            children = sources["detect_isDeblendedSource"] == 1
            sources = sources[children].copy(deep=True)
            # Clear parents, so that measurement plugins behave correctly.
            sources['parent'] = 0
            return sources.copy(deep=True)

        positives = deblend(positiveFootprints)
        negatives = deblend(negativeFootprints, negative=True)

        sources = afwTable.SourceCatalog(self.schema)
        sources.reserve(len(positives) + len(negatives))
        sources.extend(positives, deep=True)
        sources.extend(negatives, deep=True)
        return sources, makeFootprints(positives), makeFootprints(negatives)

    def _removeBadSources(self, diaSources):
        """Remove unphysical diaSources from the catalog.

        Parameters
        ----------
        diaSources : `lsst.afw.table.SourceCatalog`
            The catalog of detected sources.

        Returns
        -------
        diaSources : `lsst.afw.table.SourceCatalog`
            The updated catalog of detected sources, with any source that has a
            flag in ``config.badSourceFlags`` set removed.
        """
        selector = np.ones(len(diaSources), dtype=bool)
        for flag in self.config.badSourceFlags:
            flags = diaSources[flag]
            nBad = np.count_nonzero(flags)
            if nBad > 0:
                self.log.debug("Found %d unphysical sources with flag %s.", nBad, flag)
                selector &= ~flags
        nBadTotal = np.count_nonzero(~selector)
        self.metadata["nRemovedBadFlaggedSources"] = nBadTotal
        self.log.info("Removed %d unphysical sources.", nBadTotal)
        return diaSources[selector].copy(deep=True)

    def addSkySources(self, diaSources, mask, seed,
                      subtask=None):
        """Add sources in empty regions of the difference image
        for measuring the background.

        Parameters
        ----------
        diaSources : `lsst.afw.table.SourceCatalog`
            The catalog of detected sources.
        mask : `lsst.afw.image.Mask`
            Mask plane for determining regions where Sky sources can be added.
        seed : `int`
            Seed value to initialize the random number generator.
        """
        if subtask is None:
            subtask = self.skySources
        skySourceFootprints = subtask.run(mask=mask, seed=seed, catalog=diaSources)
        self.metadata[f"n_{subtask.getName()}"] = len(skySourceFootprints)

    def measureDiaSources(self, diaSources, science, difference, matchedTemplate):
        """Use (matched) template and science image to constrain dipole fitting.

        Parameters
        ----------
        diaSources : `lsst.afw.table.SourceCatalog`
            The catalog of detected sources.
        science : `lsst.afw.image.ExposureF`
            Science exposure that the template was subtracted from.
        difference : `lsst.afw.image.ExposureF`
            Result of subtracting template from the science image.
        matchedTemplate : `lsst.afw.image.ExposureF`
            Warped and PSF-matched template that was used produce the
            difference image.
        """
        # Ensure that the required mask planes are present
        for mp in self.config.measurement.plugins["base_PixelFlags"].masksFpAnywhere:
            difference.mask.addMaskPlane(mp)
        # Note that this may not be correct if we convolved the science image.
        # In the future we may wish to persist the matchedScience image.
        self.measurement.run(diaSources, difference, science, matchedTemplate)
        if self.config.doApCorr:
            apCorrMap = difference.getInfo().getApCorrMap()
            if apCorrMap is None:
                self.log.warning("Difference image does not have valid aperture correction; skipping.")
            else:
                self.applyApCorr.run(
                    catalog=diaSources,
                    apCorrMap=apCorrMap,
                )

    def measureForcedSources(self, diaSources, science, wcs):
        """Perform forced measurement of the diaSources on the science image.

        Parameters
        ----------
        diaSources : `lsst.afw.table.SourceCatalog`
            The catalog of detected sources.
        science : `lsst.afw.image.ExposureF`
            Science exposure that the template was subtracted from.
        wcs : `lsst.afw.geom.SkyWcs`
            Coordinate system definition (wcs) for the exposure.
        """
        # Run forced psf photometry on the PVI at the diaSource locations.
        # Copy the measured flux and error into the diaSource.
        forcedSources = self.forcedMeasurement.generateMeasCat(science, diaSources, wcs)
        self.forcedMeasurement.run(forcedSources, science, diaSources, wcs)
        mapper = afwTable.SchemaMapper(forcedSources.schema, diaSources.schema)
        mapper.addMapping(forcedSources.schema.find("base_PsfFlux_instFlux")[0],
                          "ip_diffim_forced_PsfFlux_instFlux", True)
        mapper.addMapping(forcedSources.schema.find("base_PsfFlux_instFluxErr")[0],
                          "ip_diffim_forced_PsfFlux_instFluxErr", True)
        mapper.addMapping(forcedSources.schema.find("base_PsfFlux_area")[0],
                          "ip_diffim_forced_PsfFlux_area", True)
        mapper.addMapping(forcedSources.schema.find("base_PsfFlux_flag")[0],
                          "ip_diffim_forced_PsfFlux_flag", True)
        mapper.addMapping(forcedSources.schema.find("base_PsfFlux_flag_noGoodPixels")[0],
                          "ip_diffim_forced_PsfFlux_flag_noGoodPixels", True)
        mapper.addMapping(forcedSources.schema.find("base_PsfFlux_flag_edge")[0],
                          "ip_diffim_forced_PsfFlux_flag_edge", True)
        for diaSource, forcedSource in zip(diaSources, forcedSources):
            diaSource.assign(forcedSource, mapper)

    def calculateMetrics(self, difference):
        """Add difference image QA metrics to the Task metadata.

        This may be used to produce corresponding metrics (see
        lsst.analysis.tools.tasks.diffimTaskDetectorVisitMetricAnalysis).

        Parameters
        ----------
        difference : `lsst.afw.image.Exposure`
            The target difference image to calculate metrics for.
        """
        mask = difference.mask
        badPix = (mask.array & mask.getPlaneBitMask(self.config.detection.excludeMaskPlanes)) > 0
        self.metadata["nGoodPixels"] = np.sum(~badPix)
        self.metadata["nBadPixels"] = np.sum(badPix)
        detPosPix = (mask.array & mask.getPlaneBitMask("DETECTED")) > 0
        detNegPix = (mask.array & mask.getPlaneBitMask("DETECTED_NEGATIVE")) > 0
        self.metadata["nPixelsDetectedPositive"] = np.sum(detPosPix)
        self.metadata["nPixelsDetectedNegative"] = np.sum(detNegPix)
        detPosPix &= badPix
        detNegPix &= badPix
        self.metadata["nBadPixelsDetectedPositive"] = np.sum(detPosPix)
        self.metadata["nBadPixelsDetectedNegative"] = np.sum(detNegPix)

        metricsMaskPlanes = list(mask.getMaskPlaneDict().keys())
        for maskPlane in metricsMaskPlanes:
            try:
                self.metadata["%s_mask_fraction"%maskPlane.lower()] = evaluateMaskFraction(mask, maskPlane)
            except InvalidParameterError:
                self.metadata["%s_mask_fraction"%maskPlane.lower()] = -1
                self.log.info("Unable to calculate metrics for mask plane %s: not in image"%maskPlane)

    def _runStreakMasking(self, difference):
        """Do streak masking and optionally save the resulting streak
        fit parameters in a catalog.

        Only returns non-empty streakInfo if self.config.writeStreakInfo
        is set. The difference image is binned by self.config.streakBinFactor
        (and detection is run a second time) so that regions with lower
        surface brightness streaks are more likely to fall above the
        detection threshold.

        Parameters
        ----------
        difference: `lsst.afw.image.Exposure`
            The exposure in which to search for streaks. Must have a detection
            mask.

        Returns
        -------
        streakInfo: `lsst.pipe.base.Struct`
            ``rho`` : `np.ndarray`
                Angle of detected streak.
            ``theta`` : `np.ndarray`
                Distance from center of detected streak.
            ``sigma`` : `np.ndarray`
                Width of streak profile.
            ``reducedChi2`` : `np.ndarray`
                Reduced chi2 of the best-fit streak profile.
            ``modelMaximum`` : `np.ndarray`
                Peak value of the fit line profile.
        """
        maskedImage = difference.maskedImage
        # Bin the diffim to enhance low surface brightness streaks
        binnedMaskedImage = afwMath.binImage(maskedImage,
                                             self.config.streakBinFactor,
                                             self.config.streakBinFactor)
        binnedExposure = afwImage.ExposureF(binnedMaskedImage.getBBox())
        binnedExposure.setMaskedImage(binnedMaskedImage)
        # Clear the DETECTED mask plane before streak detection
        binnedExposure.mask &= ~binnedExposure.mask.getPlaneBitMask('DETECTED')
        # Rerun detection to set the DETECTED mask plane on binnedExposure
        sigma = difference.psf.computeShape(difference.psf.getAveragePosition()).getDeterminantRadius()
        _table = afwTable.SourceTable.make(afwTable.SourceTable.makeMinimalSchema())
        self.streakDetection.run(table=_table, exposure=binnedExposure, doSmooth=True,
                                 sigma=sigma/self.config.streakBinFactor)
        binnedDetectedMaskPlane = binnedExposure.mask.array & binnedExposure.mask.getPlaneBitMask('DETECTED')
        rescaledDetectedMaskPlane = binnedDetectedMaskPlane.repeat(self.config.streakBinFactor,
                                                                   axis=0).repeat(self.config.streakBinFactor,
                                                                                  axis=1)
        # Create new version of a diffim with DETECTED based on binnedExposure
        streakMaskedImage = maskedImage.clone()
        ysize, xsize = rescaledDetectedMaskPlane.shape
        streakMaskedImage.mask.array[:ysize, :xsize] |= rescaledDetectedMaskPlane
        # Detect streaks on this new version of the diffim
        streaks = self.maskStreaks.run(streakMaskedImage)
        streakMaskPlane = streakMaskedImage.mask.array & streakMaskedImage.mask.getPlaneBitMask('STREAK')
        # Apply the new STREAK mask to the original diffim
        maskedImage.mask.array |= streakMaskPlane

        if self.config.writeStreakInfo:
            rhos = np.array([line.rho for line in streaks.lines])
            thetas = np.array([line.theta for line in streaks.lines])
            sigmas = np.array([line.sigma for line in streaks.lines])
            chi2s = np.array([line.reducedChi2 for line in streaks.lines])
            modelMaximums = np.array([line.modelMaximum for line in streaks.lines])
            streakInfo = {'rho': rhos, 'theta': thetas, 'sigma': sigmas, 'reducedChi2': chi2s,
                          'modelMaximum': modelMaximums}
        else:
            streakInfo = {'rho': np.array([]), 'theta': np.array([]), 'sigma': np.array([]),
                          'reducedChi2': np.array([]), 'modelMaximum': np.array([])}
        return pipeBase.Struct(maskedStreaks=streakInfo)


class DetectAndMeasureScoreConnections(DetectAndMeasureConnections):
    scoreExposure = pipeBase.connectionTypes.Input(
        doc="Maximum likelihood image for detection.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_scoreExp",
    )


class DetectAndMeasureScoreConfig(DetectAndMeasureConfig,
                                  pipelineConnections=DetectAndMeasureScoreConnections):
    pass


class DetectAndMeasureScoreTask(DetectAndMeasureTask):
    """Detect DIA sources using a score image,
    and measure the detections on the difference image.

    Source detection is run on the supplied score, or maximum likelihood,
    image. Note that no additional convolution will be done in this case.
    Close positive and negative detections will optionally be merged into
    dipole diaSources.
    Sky sources, or forced detections in background regions, will optionally
    be added, and the configured measurement algorithm will be run on all
    detections.
    """
    ConfigClass = DetectAndMeasureScoreConfig
    _DefaultName = "detectAndMeasureScore"

    @timeMethod
    def run(self, science, matchedTemplate, difference, scoreExposure,
            idFactory=None):
        """Detect and measure sources on a score image.

        Parameters
        ----------
        science : `lsst.afw.image.ExposureF`
            Science exposure that the template was subtracted from.
        matchedTemplate : `lsst.afw.image.ExposureF`
            Warped and PSF-matched template that was used produce the
            difference image.
        difference : `lsst.afw.image.ExposureF`
            Result of subtracting template from the science image.
        scoreExposure : `lsst.afw.image.ExposureF`
            Score or maximum likelihood difference image
        idFactory : `lsst.afw.table.IdFactory`, optional
            Generator object used to assign ids to detected sources in the
            difference image. Ids from this generator are not set until after
            deblending and merging positive/negative peaks.

        Returns
        -------
        measurementResults : `lsst.pipe.base.Struct`

            ``subtractedMeasuredExposure`` : `lsst.afw.image.ExposureF`
                Subtracted exposure with detection mask applied.
            ``diaSources``  : `lsst.afw.table.SourceCatalog`
                The catalog of detected sources.
        """
        if idFactory is None:
            idFactory = lsst.meas.base.IdGenerator().make_table_id_factory()

        self._prepareInputs(scoreExposure)

        # Don't use the idFactory until after deblend+merge, so that we aren't
        # generating ids that just get thrown away (footprint merge doesn't
        # know about past ids).
        table = afwTable.SourceTable.make(self.schema)
        results = self.detection.run(
            table=table,
            exposure=scoreExposure,
            doSmooth=False,
        )
        # Copy the detection mask from the Score image to the difference image
        difference.mask.assign(scoreExposure.mask, scoreExposure.getBBox())

        sources, positives, negatives = self._deblend(difference,
                                                      results.positive,
                                                      results.negative)

        return self.processResults(science, matchedTemplate, difference, sources, idFactory,
                                   positiveFootprints=positives, negativeFootprints=negatives)
