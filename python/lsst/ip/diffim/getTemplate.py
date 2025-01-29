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
import collections

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
                             dimensions=("instrument", "visit", "detector", "skymap"),
                             defaultTemplates={"coaddName": "goodSeeing",
                                               "warpTypeSuffix": "",
                                               "fakesType": ""}):
    bbox = pipeBase.connectionTypes.Input(
        doc="Bounding box of exposure to determine the geometry of the output template.",
        name="{fakesType}calexp.bbox",
        storageClass="Box2I",
        dimensions=("instrument", "visit", "detector"),
    )
    wcs = pipeBase.connectionTypes.Input(
        doc="WCS of the exposure that we will construct the template for.",
        name="{fakesType}calexp.wcs",
        storageClass="Wcs",
        dimensions=("instrument", "visit", "detector"),
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Geometry of the tracts and patches that the coadds are defined on.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap", ),
        storageClass="SkyMap",
    )
    coaddExposures = pipeBase.connectionTypes.Input(
        doc="Coadds that may overlap the desired region, as possible inputs to the template."
            " Will be restricted to those that directly overlap the projected bounding box.",
        dimensions=("tract", "patch", "skymap", "band"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Coadd{warpTypeSuffix}",
        multiple=True,
        deferLoad=True
    )

    template = pipeBase.connectionTypes.Output(
        doc="Warped template, pixel matched to the bounding box and WCS.",
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
        # Use a smaller cache: per SeparableKernel.computeCache, this should
        # give a warping error of a fraction of a count (these must match).
        self.warp.cacheSize = 100000
        self.coaddPsf.cacheSize = self.warp.cacheSize
        # The WCS for LSST should be smoothly varying, so we can use a longer
        # interpolation length for WCS evaluations.
        self.warp.interpLength = 100
        self.warp.warpingKernelName = 'lanczos3'
        self.coaddPsf.warpingKernelName = self.warp.warpingKernelName


class GetTemplateTask(pipeBase.PipelineTask):
    ConfigClass = GetTemplateConfig
    _DefaultName = "getTemplate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warper = afwMath.Warper.fromConfig(self.config.warp)
        self.schema = afwTable.ExposureTable.makeMinimalSchema()
        self.schema.addField('tract', type=np.int32, doc='Which tract this exposure came from.')
        self.schema.addField('patch', type=np.int32, doc='Which patch in the tract this exposure came from.')
        self.schema.addField('weight', type=float,
                             doc='Weight for each exposure, used to make the CoaddPsf; should always be 1.')

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        results = self.getOverlappingExposures(inputs)
        del inputs["skyMap"]  # Only needed for the above.
        inputs["coaddExposures"] = results.coaddExposures
        inputs["dataIds"] = results.dataIds
        inputs["physical_filter"] = butlerQC.quantum.dataId["physical_filter"]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def getOverlappingExposures(self, inputs):
        """Return a data structure containing the coadds that overlap the
        specified bbox projected onto the sky, and a corresponding data
        structure of their dataIds.
        These are the appropriate inputs to this task's `run` method.

        The spatial index in the butler registry has generous padding and often
        supplies patches near, but not directly overlapping the desired region.
        This method filters the inputs so that `run` does not have to read in
        all possibly-matching coadd exposures.

        Parameters
        ----------
        inputs : `dict` of task Inputs, containing:
            - coaddExposures : `list` \
                              [`lsst.daf.butler.DeferredDatasetHandle` of \
                               `lsst.afw.image.Exposure`]
                Data references to exposures that might overlap the desired
                region.
            - bbox : `lsst.geom.Box2I`
                Template bounding box of the pixel geometry onto which the
                coaddExposures will be resampled.
            - skyMap : `lsst.skymap.SkyMap`
                Geometry of the tracts and patches the coadds are defined on.
            - wcs : `lsst.afw.geom.SkyWcs`
                Template WCS onto which the coadds will be resampled.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
           A struct with attributes:

           ``coaddExposures``
               Dict of coadd exposures that overlap the projected bbox,
               indexed on tract id
               (`dict` [`int`, `list` [`lsst.afw.image.Exposure`] ]).
           ``dataIds``
               Dict of data IDs of the coadd exposures that overlap the
               projected bbox, indexed on tract id
               (`dict` [`int`, `list [`lsst.daf.butler.DataCoordinate`] ]).

        Raises
        ------
        NoWorkFound
            Raised if no patches overlap the input detector bbox, or the input
            WCS is None.
        """
        if (wcs := inputs['wcs']) is None:
            raise pipeBase.NoWorkFound("Exposure has no WCS; cannot create a template.")

        # Exposure's validPolygon would be more accurate
        detectorPolygon = geom.Box2D(inputs['bbox'])
        overlappingArea = 0
        coaddExposures = collections.defaultdict(list)
        dataIds = collections.defaultdict(list)

        skymap = inputs['skyMap']
        for coaddRef in inputs['coaddExposures']:
            dataId = coaddRef.dataId
            patchWcs = skymap[dataId['tract']].getWcs()
            patchBBox = skymap[dataId['tract']][dataId['patch']].getOuterBBox()
            patchCorners = patchWcs.pixelToSky(geom.Box2D(patchBBox).getCorners())
            patchPolygon = afwGeom.Polygon(wcs.skyToPixel(patchCorners))
            if patchPolygon.intersection(detectorPolygon):
                overlappingArea += patchPolygon.intersectionSingle(detectorPolygon).calculateArea()
                self.log.info("Using template input tract=%s, patch=%s", dataId['tract'], dataId['patch'])
                coaddExposures[dataId['tract']].append(coaddRef.get())
                dataIds[dataId['tract']].append(dataId)

        if not overlappingArea:
            raise pipeBase.NoWorkFound('No patches overlap detector')

        return pipeBase.Struct(coaddExposures=coaddExposures,
                               dataIds=dataIds)

    @timeMethod
    def run(self, coaddExposures, bbox, wcs, dataIds, physical_filter):
        """Warp coadds from multiple tracts and patches to form a template to
        subtract from a science image.

        Tract and patch overlap regions are combined by a variance-weighted
        average, and the variance planes are combined with the same weights,
        not added in quadrature; the overlap regions are not statistically
        independent, because they're derived from the same original data.
        The PSF on the template is created by combining the CoaddPsf on each
        template image into a meta-CoaddPsf.

        Parameters
        ----------
        coaddExposures : `dict` [`int`, `list` [`lsst.afw.image.Exposure`]]
            Coadds to be mosaicked, indexed on tract id.
        bbox : `lsst.geom.Box2I`
            Template Bounding box of the detector geometry onto which to
            resample the ``coaddExposures``. Modified in-place to include the
            template border.
        wcs : `lsst.afw.geom.SkyWcs`
            Template WCS onto which to resample the ``coaddExposures``.
        dataIds : `dict` [`int`, `list` [`lsst.daf.butler.DataCoordinate`]]
            Record of the tract and patch of each coaddExposure, indexed on
            tract id.
        physical_filter : `str`
            Physical filter of the science image.

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
        """
        band, photoCalib = self._checkInputs(dataIds, coaddExposures)

        bbox.grow(self.config.templateBorderSize)

        warped = {}
        catalogs = []
        for tract in coaddExposures:
            maskedImages, catalog, totalBox = self._makeExposureCatalog(coaddExposures[tract],
                                                                        dataIds[tract])
            # Combine images from individual patches together.
            unwarped = self._merge(maskedImages, totalBox, catalog[0].wcs)
            potentialInput = self.warper.warpExposure(wcs, unwarped, destBBox=bbox)
            if not np.any(np.isfinite(potentialInput.image.array)):
                self.log.info("No overlap from coadds in tract %s; not including in output.", tract)
                continue
            catalogs.append(catalog)
            warped[tract] = potentialInput
            warped[tract].setWcs(wcs)

        if len(warped) == 0:
            raise pipeBase.NoWorkFound("No patches found to overlap science exposure.")
        template = self._merge([x.maskedImage for x in warped.values()], bbox, wcs)

        # Make a single catalog containing all the inputs that were accepted.
        catalog = afwTable.ExposureCatalog(self.schema)
        catalog.reserve(sum([len(c) for c in catalogs]))
        for c in catalogs:
            catalog.extend(c)

        template.setPsf(self._makePsf(template, catalog, wcs))
        template.setFilter(afwImage.FilterLabel(band, physical_filter))
        template.setPhotoCalib(photoCalib)
        return pipeBase.Struct(template=template)

    @staticmethod
    def _checkInputs(dataIds, coaddExposures):
        """Check that the all the dataIds are from the same band and that
        the exposures all have the same photometric calibration.

        Parameters
        ----------
        dataIds : `dict` [`int`, `list` [`lsst.daf.butler.DataCoordinate`]]
            Record of the tract and patch of each coaddExposure.
        coaddExposures : `dict` [`int`, `list` [`lsst.afw.image.Exposure`]]
            Coadds to be mosaicked.

        Returns
        -------
        band : `str`
            Filter band of all the input exposures.
        photoCalib : `lsst.afw.image.PhotoCalib`
            Photometric calibration of all of the input exposures.

        Raises
        ------
        RuntimeError
            Raised if the bands or calibrations of the input exposures are not
            all the same.
        """
        bands = set(dataId["band"] for tract in dataIds for dataId in dataIds[tract])
        if len(bands) > 1:
            raise RuntimeError(f"GetTemplateTask called with multiple bands: {bands}")
        band = bands.pop()
        photoCalibs = [exposure.photoCalib for exposures in coaddExposures.values() for exposure in exposures]
        if not all([photoCalibs[0] == x for x in photoCalibs]):
            msg = f"GetTemplateTask called with exposures with different photoCalibs: {photoCalibs}"
            raise RuntimeError(msg)
        photoCalib = photoCalibs[0]
        return band, photoCalib

    def _makeExposureCatalog(self, exposures, dataIds):
        """Make an exposure catalog for one tract.

        Parameters
        ----------
        exposures : `list` [`lsst.afw.image.Exposuref`]
            Exposures to include in the catalog.
        dataIds : `list` [`lsst.daf.butler.DataCoordinate`]
            Data ids of each of the included exposures; must have "tract" and
            "patch" entries.

        Returns
        -------
        images : `list` [`lsst.afw.image.MaskedImage`]
            MaskedImages of each of the input exposures, for warping.
        catalog : `lsst.afw.table.ExposureCatalog`
            Catalog of metadata for each exposure
        totalBox : `lsst.geom.Box2I`
            The union of the bounding boxes of all the input exposures.
        """
        catalog = afwTable.ExposureCatalog(self.schema)
        catalog.reserve(len(exposures))
        images = [exposure.maskedImage for exposure in exposures]
        totalBox = geom.Box2I()
        for coadd, dataId in zip(exposures, dataIds):
            totalBox = totalBox.expandedTo(coadd.getBBox())
            record = catalog.addNew()
            record.setPsf(coadd.psf)
            record.setWcs(coadd.wcs)
            record.setPhotoCalib(coadd.photoCalib)
            record.setBBox(coadd.getBBox())
            record.setValidPolygon(afwGeom.Polygon(geom.Box2D(coadd.getBBox()).getCorners()))
            record.set("tract", dataId["tract"])
            record.set("patch", dataId["patch"])
            # Weight is used by CoaddPsf, but the PSFs from overlapping patches
            # should be very similar, so this value mostly shouldn't matter.
            record.set("weight", 1)

        return images, catalog, totalBox

    def _merge(self, maskedImages, bbox, wcs):
        """Merge the images that came from one tract into one larger image,
        ignoring NaN pixels and non-finite variance pixels from individual
        exposures.

        Parameters
        ----------
        maskedImages : `list` [`lsst.afw.image.MaskedImage`]
            Images to be merged into one larger bounding box.
        bbox : `lsst.geom.Box2I`
            Bounding box defining the image to merge into.
        wcs : `lsst.afw.geom.SkyWcs`
            WCS of all of the input images to set on the output image.

        Returns
        -------
        merged : `lsst.afw.image.MaskedImage`
            Merged image with all of the inputs at their respective bbox
            positions.
        """
        merged = afwImage.ExposureF(bbox, wcs)
        weights = afwImage.ImageF(bbox)
        for maskedImage in maskedImages:
            weight = afwImage.ImageF(maskedImage.variance.array**(-0.5))
            bad = np.isnan(maskedImage.image.array) | ~np.isfinite(maskedImage.variance.array)
            # Note that modifying the patch MaskedImage in place is fine;
            # we're throwing it away at the end anyway.
            maskedImage.image.array[bad] = 0.0
            maskedImage.variance.array[bad] = 0.0
            # Reset mask, too, since these pixels don't contribute to sum.
            maskedImage.mask.array[bad] = 0
            # Cannot use `merged.maskedImage *= weight` because that operator
            # multiplies the variance by the weight twice; in this case
            # `weight` are the exact values we want to scale by.
            maskedImage.image *= weight
            maskedImage.variance *= weight
            merged.maskedImage[maskedImage.getBBox()] += maskedImage
            # Clear the NaNs to ensure that areas missing from this input are
            # masked with NO_DATA after the loop.
            weight.array[np.isnan(weight.array)] = 0
            weights[maskedImage.getBBox()] += weight
        # Cannot use `merged.maskedImage /= weights` because that operator
        # divides the variance by the weight twice; in this case `weights` are
        # the exact values we want to scale by.
        merged.image /= weights
        merged.variance /= weights
        merged.mask.array |= merged.mask.getPlaneBitMask("NO_DATA") * (weights.array == 0)

        return merged

    def _makePsf(self, template, catalog, wcs):
        """Return a PSF containing the PSF at each of the input regions.

        Note that although this includes all the exposures from the catalog,
        the PSF knows which part of the template the inputs came from, so when
        evaluated at a given position it will not include inputs that never
        went in to those pixels.

        Parameters
        ----------
        template : `lsst.afw.image.Exposure`
            Generated template the PSF is for.
        catalog : `lsst.afw.table.ExposureCatalog`
            Catalog of exposures that went into the template that contains all
            of the input PSFs.
        wcs : `lsst.afw.geom.SkyWcs`
            WCS of the template, to warp the PSFs to.

        Returns
        -------
        coaddPsf : `lsst.meas.algorithms.CoaddPsf`
            The meta-psf constructed from all of the input catalogs.
        """
        # CoaddPsf centroid not only must overlap image, but must overlap the
        # part of image with data. Use centroid of region with data.
        boolmask = template.mask.array & template.mask.getPlaneBitMask('NO_DATA') == 0
        maskx = afwImage.makeMaskFromArray(boolmask.astype(afwImage.MaskPixel))
        centerCoord = afwGeom.SpanSet.fromMask(maskx, 1).computeCentroid()

        ctrl = self.config.coaddPsf.makeControl()
        coaddPsf = CoaddPsf(catalog, wcs, centerCoord, ctrl.warpingKernelName, ctrl.cacheSize)
        return coaddPsf


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
            - coaddExposureRefs : `list` \
                                  [`lsst.daf.butler.DeferredDatasetHandle` of \
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
        coaddExposures : `list` [`lsst.afw.image.Exposure`]
            Coadd exposures that overlap the detector.
        """
        coaddExposures = []
        for tract in patchList:
            for patch in set(patchList[tract]):
                coaddRefList = [coaddRef for coaddRef in coaddRefs
                                if _selectDataRef(coaddRef, tract, patch)]

                dcrModel = DcrModel.fromQuantum(coaddRefList,
                                                self.config.effectiveWavelength,
                                                self.config.bandwidth,
                                                self.config.numSubfilters)
                coaddExposures.append(dcrModel.buildMatchedExposure(visitInfo=visitInfo))
        return coaddExposures


def _selectDataRef(coaddRef, tract, patch):
    condition = (coaddRef.dataId['tract'] == tract) & (coaddRef.dataId['patch'] == patch)
    return condition
