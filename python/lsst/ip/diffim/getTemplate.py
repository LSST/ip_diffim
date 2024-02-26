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

import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import BaseSkyMap
from lsst.ip.diffim.dcrModel import DcrModel
from lsst.meas.algorithms import CoaddPsf, CoaddPsfConfig
from lsst.utils.timer import timeMethod

__all__ = ["GetTemplateTask", "GetTemplateConfig",
           "GetDcrTemplateTask", "GetDcrTemplateConfig"]


class GetTemplateConnections(pipeBase.PipelineTaskConnections,
                             dimensions=("instrument", "visit", "detector"),
                             defaultTemplates={"coaddName": "goodSeeing",
                                               "warpTypeSuffix": "",
                                               "fakesType": ""}):
    bbox = pipeBase.connectionTypes.Input(
        doc="BBoxes of calexp used determine geometry of output template",
        name="{fakesType}calexp.bbox",
        storageClass="Box2I",
        dimensions=("instrument", "visit", "detector"),
    )
    wcs = pipeBase.connectionTypes.Input(
        doc="WCS of the calexp that we want to fetch the template for",
        name="{fakesType}calexp.wcs",
        storageClass="Wcs",
        dimensions=("instrument", "visit", "detector"),
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for template exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap", ),
        storageClass="SkyMap",
    )
    # TODO DM-31292: Add option to use global external wcs from jointcal
    # Needed for DRP HSC
    coaddExposures = pipeBase.connectionTypes.Input(
        doc="Input template to match and subtract from the exposure",
        dimensions=("tract", "patch", "skymap", "band"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Coadd{warpTypeSuffix}",
        multiple=True,
        deferLoad=True
    )
    template = pipeBase.connectionTypes.Output(
        doc="Warped template used to create `subtractedExposure`.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_templateExp{warpTypeSuffix}",
    )


class GetTemplateConfig(pipeBase.PipelineTaskConfig,
                        pipelineConnections=GetTemplateConnections):
    templateBorderSize = pexConfig.Field(
        dtype=int,
        default=20,
        doc="Number of pixels to grow the requested template image to account for warping"
    )
    warp = pexConfig.ConfigField(
        dtype=afwMath.Warper.ConfigClass,
        doc="warper configuration",
    )
    coaddPsf = pexConfig.ConfigField(
        doc="Configuration for CoaddPsf",
        dtype=CoaddPsfConfig,
    )

    def setDefaults(self):
        self.warp.warpingKernelName = 'lanczos5'
        self.coaddPsf.warpingKernelName = 'lanczos5'


class GetTemplateTask(pipeBase.PipelineTask):
    ConfigClass = GetTemplateConfig
    _DefaultName = "getTemplate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warper = afwMath.Warper.fromConfig(self.config.warp)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Read in all inputs.
        inputs = butlerQC.get(inputRefs)
        results = self.getOverlappingExposures(inputs)
        inputs["coaddExposures"] = results.coaddExposures
        inputs["dataIds"] = results.dataIds
        inputs["physical_filter"] = butlerQC.quantum.dataId["physical_filter"]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def getOverlappingExposures(self, inputs):
        """Return lists of coadds and their corresponding dataIds that overlap
        the detector.

        The spatial index in the registry has generous padding and often
        supplies patches near, but not directly overlapping the detector.
        Filters inputs so that we don't have to read in all input coadds.

        Parameters
        ----------
        inputs : `dict` of task Inputs, containing:
            - coaddExposureRefs : `list`
                                [`lsst.daf.butler.DeferredDatasetHandle` of
                                `lsst.afw.image.Exposure`]
                Data references to exposures that might overlap the detector.
            - bbox : `lsst.geom.Box2I`
                Template Bounding box of the detector geometry onto which to
                resample the coaddExposures.
            - skyMap : `lsst.skymap.SkyMap`
                Input definition of geometry/bbox and projection/wcs for
                template exposures.
            - wcs : `lsst.afw.geom.SkyWcs`
                Template WCS onto which to resample the coaddExposures.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
           A struct with attributes:

           ``coaddExposures``
               List of Coadd exposures that overlap the detector (`list`
               [`lsst.afw.image.Exposure`]).
           ``dataIds``
               List of data IDs of the coadd exposures that overlap the
               detector (`list` [`lsst.daf.butler.DataCoordinate`]).

        Raises
        ------
        NoWorkFound
            Raised if no patches overlap the input detector bbox.
        """
        # Check that the patches actually overlap the detector
        # Exposure's validPolygon would be more accurate
        detectorPolygon = geom.Box2D(inputs['bbox'])
        overlappingArea = 0
        coaddExposureList = []
        dataIds = []
        for coaddRef in inputs['coaddExposures']:
            dataId = coaddRef.dataId
            patchWcs = inputs['skyMap'][dataId['tract']].getWcs()
            patchBBox = inputs['skyMap'][dataId['tract']][dataId['patch']].getOuterBBox()
            patchCorners = patchWcs.pixelToSky(geom.Box2D(patchBBox).getCorners())
            inputsWcs = inputs['wcs']
            if inputsWcs is not None:
                patchPolygon = afwGeom.Polygon(inputsWcs.skyToPixel(patchCorners))
                if patchPolygon.intersection(detectorPolygon):
                    overlappingArea += patchPolygon.intersectionSingle(detectorPolygon).calculateArea()
                    self.log.info("Using template input tract=%s, patch=%s" %
                                  (dataId['tract'], dataId['patch']))
                    coaddExposureList.append(coaddRef.get())
                    dataIds.append(dataId)
            else:
                self.log.info("Exposure has no WCS, so cannot create associated template.")

        if not overlappingArea:
            raise pipeBase.NoWorkFound('No patches overlap detector')

        return pipeBase.Struct(coaddExposures=coaddExposureList,
                               dataIds=dataIds)

    @timeMethod
    def run(self, coaddExposures, bbox, wcs, dataIds, physical_filter=None, **kwargs):
        """Warp coadds from multiple tracts to form a template for image diff.

        Where the tracts overlap, the resulting template image is averaged.
        The PSF on the template is created by combining the CoaddPsf on each
        template image into a meta-CoaddPsf.

        Parameters
        ----------
        coaddExposures : `list` [`lsst.afw.image.Exposure`]
            Coadds to be mosaicked.
        bbox : `lsst.geom.Box2I`
            Template Bounding box of the detector geometry onto which to
            resample the ``coaddExposures``.
        wcs : `lsst.afw.geom.SkyWcs`
            Template WCS onto which to resample the ``coaddExposures``.
        dataIds : `list` [`lsst.daf.butler.DataCoordinate`]
            Record of the tract and patch of each coaddExposure.
        physical_filter : `str`, optional
            The physical filter of the science image.
        **kwargs
            Any additional keyword parameters.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
           A struct with attributes:

           ``template``
               A template coadd exposure assembled out of patches
               (`lsst.afw.image.ExposureF`).

        Raises
        ------
        NoWorkFound
            If no coadds are found with sufficient un-masked pixels.
        RuntimeError
            If the PSF of the template can't be calculated.
        """
        # Table for CoaddPSF
        tractsSchema = afwTable.ExposureTable.makeMinimalSchema()
        tractKey = tractsSchema.addField('tract', type=np.int32, doc='Which tract')
        patchKey = tractsSchema.addField('patch', type=np.int32, doc='Which patch')
        weightKey = tractsSchema.addField('weight', type=float, doc='Weight for each tract, should be 1')
        tractsCatalog = afwTable.ExposureCatalog(tractsSchema)

        finalWcs = wcs
        bbox.grow(self.config.templateBorderSize)
        finalBBox = bbox

        nPatchesFound = 0
        maskedImageList = []
        weightList = []

        for coaddExposure, dataId in zip(coaddExposures, dataIds):

            # warp to detector WCS
            warped = self.warper.warpExposure(finalWcs, coaddExposure, maxBBox=finalBBox)

            # Check if warped image is viable
            if not np.any(np.isfinite(warped.image.array)):
                self.log.info("No overlap for warped %s. Skipping" % dataId)
                continue

            exp = afwImage.ExposureF(finalBBox, finalWcs)
            exp.maskedImage.set(np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan)
            exp.maskedImage.assign(warped.maskedImage, warped.getBBox())

            maskedImageList.append(exp.maskedImage)
            weightList.append(1)
            record = tractsCatalog.addNew()
            record.setPsf(coaddExposure.getPsf())
            record.setWcs(coaddExposure.getWcs())
            record.setPhotoCalib(coaddExposure.getPhotoCalib())
            record.setBBox(coaddExposure.getBBox())
            record.setValidPolygon(afwGeom.Polygon(geom.Box2D(coaddExposure.getBBox()).getCorners()))
            record.set(tractKey, dataId['tract'])
            record.set(patchKey, dataId['patch'])
            record.set(weightKey, 1.)
            nPatchesFound += 1

        if nPatchesFound == 0:
            raise pipeBase.NoWorkFound("No patches found to overlap detector")

        # Combine images from individual patches together
        statsFlags = afwMath.stringToStatisticsProperty('MEAN')
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNanSafe(True)
        statsCtrl.setWeighted(True)
        statsCtrl.setCalcErrorMosaicMode(True)

        templateExposure = afwImage.ExposureF(finalBBox, finalWcs)
        templateExposure.maskedImage.set(np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan)
        xy0 = templateExposure.getXY0()
        # Do not mask any values
        templateExposure.maskedImage = afwMath.statisticsStack(maskedImageList, statsFlags, statsCtrl,
                                                               weightList, clipped=0, maskMap=[])
        templateExposure.maskedImage.setXY0(xy0)

        # CoaddPsf centroid not only must overlap image, but must overlap the
        # part of image with data. Use centroid of region with data.
        boolmask = templateExposure.mask.array & templateExposure.mask.getPlaneBitMask('NO_DATA') == 0
        maskx = afwImage.makeMaskFromArray(boolmask.astype(afwImage.MaskPixel))
        centerCoord = afwGeom.SpanSet.fromMask(maskx, 1).computeCentroid()

        ctrl = self.config.coaddPsf.makeControl()
        coaddPsf = CoaddPsf(tractsCatalog, finalWcs, centerCoord, ctrl.warpingKernelName, ctrl.cacheSize)
        if coaddPsf is None:
            raise RuntimeError("CoaddPsf could not be constructed")

        templateExposure.setPsf(coaddPsf)
        # Coadds do not have a physical filter, so fetch it from the butler to prevent downstream warnings.
        if physical_filter is None:
            filterLabel = coaddExposure.getFilter()
        else:
            filterLabel = afwImage.FilterLabel(dataId['band'], physical_filter)
        templateExposure.setFilter(filterLabel)
        templateExposure.setPhotoCalib(coaddExposure.getPhotoCalib())
        return pipeBase.Struct(template=templateExposure)


class GetDcrTemplateConnections(GetTemplateConnections,
                                dimensions=("instrument", "visit", "detector", "skymap"),
                                defaultTemplates={"coaddName": "dcr",
                                                  "warpTypeSuffix": "",
                                                  "fakesType": ""}):
    visitInfo = pipeBase.connectionTypes.Input(
        doc="VisitInfo of calexp used to determine observing conditions.",
        name="{fakesType}calexp.visitInfo",
        storageClass="VisitInfo",
        dimensions=("instrument", "visit", "detector"),
    )
    dcrCoadds = pipeBase.connectionTypes.Input(
        doc="Input DCR template to match and subtract from the exposure",
        name="{fakesType}dcrCoadd{warpTypeSuffix}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band", "subfilter"),
        multiple=True,
        deferLoad=True
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        self.inputs.remove("coaddExposures")


class GetDcrTemplateConfig(GetTemplateConfig,
                           pipelineConnections=GetDcrTemplateConnections):
    numSubfilters = pexConfig.Field(
        doc="Number of subfilters in the DcrCoadd.",
        dtype=int,
        default=3,
    )
    effectiveWavelength = pexConfig.Field(
        doc="Effective wavelength of the filter.",
        optional=False,
        dtype=float,
    )
    bandwidth = pexConfig.Field(
        doc="Bandwidth of the physical filter.",
        optional=False,
        dtype=float,
    )

    def validate(self):
        if self.effectiveWavelength is None or self.bandwidth is None:
            raise ValueError("The effective wavelength and bandwidth of the physical filter "
                             "must be set in the getTemplate config for DCR coadds. "
                             "Required until transmission curves are used in DM-13668.")


class GetDcrTemplateTask(GetTemplateTask):
    ConfigClass = GetDcrTemplateConfig
    _DefaultName = "getDcrTemplate"

    def getOverlappingExposures(self, inputs):
        """Return lists of coadds and their corresponding dataIds that overlap
        the detector.

        The spatial index in the registry has generous padding and often
        supplies patches near, but not directly overlapping the detector.
        Filters inputs so that we don't have to read in all input coadds.

        Parameters
        ----------
        inputs : `dict` of task Inputs, containing:
            - coaddExposureRefs : `list`
                                  [`lsst.daf.butler.DeferredDatasetHandle` of
                                  `lsst.afw.image.Exposure`]
                Data references to exposures that might overlap the detector.
            - bbox : `lsst.geom.Box2I`
                Template Bounding box of the detector geometry onto which to
                resample the coaddExposures.
            - skyMap : `lsst.skymap.SkyMap`
                Input definition of geometry/bbox and projection/wcs for
                template exposures.
            - wcs : `lsst.afw.geom.SkyWcs`
                Template WCS onto which to resample the coaddExposures.
            - visitInfo : `lsst.afw.image.VisitInfo`
                Metadata for the science image.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
           A struct with attibutes:

           ``coaddExposures``
               Coadd exposures that overlap the detector (`list`
               [`lsst.afw.image.Exposure`]).
           ``dataIds``
               Data IDs of the coadd exposures that overlap the detector
               (`list` [`lsst.daf.butler.DataCoordinate`]).

        Raises
        ------
        NoWorkFound
            Raised if no patches overlatp the input detector bbox.
        """
        # Check that the patches actually overlap the detector
        # Exposure's validPolygon would be more accurate
        detectorPolygon = geom.Box2D(inputs["bbox"])
        overlappingArea = 0
        coaddExposureRefList = []
        dataIds = []
        patchList = dict()
        for coaddRef in inputs["dcrCoadds"]:
            dataId = coaddRef.dataId
            patchWcs = inputs["skyMap"][dataId['tract']].getWcs()
            patchBBox = inputs["skyMap"][dataId['tract']][dataId['patch']].getOuterBBox()
            patchCorners = patchWcs.pixelToSky(geom.Box2D(patchBBox).getCorners())
            patchPolygon = afwGeom.Polygon(inputs["wcs"].skyToPixel(patchCorners))
            if patchPolygon.intersection(detectorPolygon):
                overlappingArea += patchPolygon.intersectionSingle(detectorPolygon).calculateArea()
                self.log.info("Using template input tract=%s, patch=%s, subfilter=%s" %
                              (dataId['tract'], dataId['patch'], dataId["subfilter"]))
                coaddExposureRefList.append(coaddRef)
                if dataId['tract'] in patchList:
                    patchList[dataId['tract']].append(dataId['patch'])
                else:
                    patchList[dataId['tract']] = [dataId['patch'], ]
                dataIds.append(dataId)

        if not overlappingArea:
            raise pipeBase.NoWorkFound('No patches overlap detector')

        self.checkPatchList(patchList)

        coaddExposures = self.getDcrModel(patchList, inputs['dcrCoadds'], inputs['visitInfo'])
        return pipeBase.Struct(coaddExposures=coaddExposures,
                               dataIds=dataIds)

    def checkPatchList(self, patchList):
        """Check that all of the DcrModel subfilters are present for each
        patch.

        Parameters
        ----------
        patchList : `dict`
            Dict of the patches containing valid data for each tract.

        Raises
        ------
        RuntimeError
            If the number of exposures found for a patch does not match the
            number of subfilters.
        """
        for tract in patchList:
            for patch in set(patchList[tract]):
                if patchList[tract].count(patch) != self.config.numSubfilters:
                    raise RuntimeError("Invalid number of DcrModel subfilters found: %d vs %d expected",
                                       patchList[tract].count(patch), self.config.numSubfilters)

    def getDcrModel(self, patchList, coaddRefs, visitInfo):
        """Build DCR-matched coadds from a list of exposure references.

        Parameters
        ----------
        patchList : `dict`
            Dict of the patches containing valid data for each tract.
        coaddRefs : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Data references to `~lsst.afw.image.Exposure` representing
            DcrModels that overlap the detector.
        visitInfo : `lsst.afw.image.VisitInfo`
            Metadata for the science image.

        Returns
        -------
        coaddExposureList : `list` [`lsst.afw.image.Exposure`]
            Coadd exposures that overlap the detector.
        """
        coaddExposureList = []
        for tract in patchList:
            for patch in set(patchList[tract]):
                coaddRefList = [coaddRef for coaddRef in coaddRefs
                                if _selectDataRef(coaddRef, tract, patch)]

                dcrModel = DcrModel.fromQuantum(coaddRefList,
                                                self.config.effectiveWavelength,
                                                self.config.bandwidth,
                                                self.config.numSubfilters)
                coaddExposureList.append(dcrModel.buildMatchedExposure(visitInfo=visitInfo))
        return coaddExposureList


def _selectDataRef(coaddRef, tract, patch):
    condition = (coaddRef.dataId['tract'] == tract) & (coaddRef.dataId['patch'] == patch)
    return condition
