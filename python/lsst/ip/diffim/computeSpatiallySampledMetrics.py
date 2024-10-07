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

import lsst.geom

import lsst.afw.table as afwTable
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from lsst.ip.diffim.utils import getPsfFwhm, angleMean, evaluateMaskFraction, getKernelCenterDisplacement
from lsst.meas.algorithms import SkyObjectsTask
from lsst.pex.exceptions import InvalidParameterError, RangeError
from lsst.utils.timer import timeMethod

import lsst.utils

__all__ = ["SpatiallySampledMetricsConfig", "SpatiallySampledMetricsTask"]


class SpatiallySampledMetricsConnections(pipeBase.PipelineTaskConnections,
                                         dimensions=("instrument", "visit", "detector"),
                                         defaultTemplates={"coaddName": "deep",
                                                           "warpTypeSuffix": "",
                                                           "fakesType": ""}):
    science = pipeBase.connectionTypes.Input(
        doc="Input science exposure.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}initial_pvi"
    )
    matchedTemplate = pipeBase.connectionTypes.Input(
        doc="Warped and PSF-matched template used to create the difference image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_matchedExp",
    )
    template = pipeBase.connectionTypes.Input(
        doc="Warped and not PSF-matched template used to create the difference image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_templateExp",
    )
    difference = pipeBase.connectionTypes.Input(
        doc="Difference image with detection mask plane filled in.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_differenceExp",
    )
    diaSources = pipeBase.connectionTypes.Input(
        doc="Filtered diaSources on the difference image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        name="{fakesType}{coaddName}Diff_candidateDiaSrc",
    )
    psfMatchingKernel = pipeBase.connectionTypes.Input(
        doc="Kernel used to PSF match the science and template images.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="MatchingKernel",
        name="{fakesType}{coaddName}Diff_psfMatchKernel",
    )
    spatiallySampledMetrics = pipeBase.connectionTypes.Output(
        doc="Summary metrics computed at randomized locations.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ArrowAstropy",
        name="{fakesType}{coaddName}Diff_spatiallySampledMetrics",
    )


class SpatiallySampledMetricsConfig(pipeBase.PipelineTaskConfig,
                                    pipelineConnections=SpatiallySampledMetricsConnections):
    """Config for SpatiallySampledMetricsTask
    """
    metricsMaskPlanes = lsst.pex.config.ListField(
        dtype=str,
        doc="List of mask planes to include in metrics",
        default=('BAD', 'CLIPPED', 'CR', 'DETECTED', 'DETECTED_NEGATIVE', 'EDGE',
                 'INEXACT_PSF', 'INJECTED', 'INJECTED_TEMPLATE', 'INTRP', 'NOT_DEBLENDED',
                 'NO_DATA', 'REJECTED', 'SAT', 'SAT_TEMPLATE', 'SENSOR_EDGE', 'STREAK', 'SUSPECT',
                 'UNMASKEDNAN',
                 ),
    )
    metricSources = pexConfig.ConfigurableField(
        target=SkyObjectsTask,
        doc="Generate QA metric sources",
    )

    def setDefaults(self):
        self.metricSources.avoidMask = ["NO_DATA", "EDGE"]


class SpatiallySampledMetricsTask(lsst.pipe.base.PipelineTask):
    """Detect and measure sources on a difference image.
    """
    ConfigClass = SpatiallySampledMetricsConfig
    _DefaultName = "spatiallySampledMetrics"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.makeSubtask("metricSources")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema.addField(
            "x", "F",
            "X location of the metric evaluation.",
            units="pixel")
        self.schema.addField(
            "y", "F",
            "Y location of the metric evaluation.",
            units="pixel")
        self.metricSources.skySourceKey = self.schema.addField("sky_source", type="Flag",
                                                               doc="Metric evaluation objects.")
        self.schema.addField(
            "source_density", "F",
            "Density of diaSources at location.",
            units="count/degree^2")
        self.schema.addField(
            "dipole_density", "F",
            "Density of dipoles at location.",
            units="count/degree^2")
        self.schema.addField(
            "dipole_direction", "F",
            "Mean dipole orientation.",
            units="radian")
        self.schema.addField(
            "dipole_separation", "F",
            "Mean dipole separation.",
            units="pixel")
        self.schema.addField(
            "template_value", "F",
            "Median of template at location.",
            units="nJy")
        self.schema.addField(
            "science_value", "F",
            "Median of science at location.",
            units="nJy")
        self.schema.addField(
            "diffim_value", "F",
            "Median of diffim at location.",
            units="nJy")
        self.schema.addField(
            "science_psfSize", "F",
            "Width of the science image PSF at location.",
            units="pixel")
        self.schema.addField(
            "template_psfSize", "F",
            "Width of the template image PSF at location.",
            units="pixel")
        for maskPlane in self.config.metricsMaskPlanes:
            self.schema.addField(
                "%s_mask_fraction"%maskPlane.lower(), "F",
                "Fraction of pixels with %s mask"%maskPlane
            )
        self.schema.addField(
            "psfMatchingKernel_sum", "F",
            "PSF matching kernel sum at location.")
        self.schema.addField(
            "psfMatchingKernel_dx", "F",
            "PSF matching kernel centroid offset in x at location.",
            units="pixel")
        self.schema.addField(
            "psfMatchingKernel_dy", "F",
            "PSF matching kernel centroid offset in y at location.",
            units="pixel")
        self.schema.addField(
            "psfMatchingKernel_length", "F",
            "PSF matching kernel centroid offset module.",
            units="arcsecond")
        self.schema.addField(
            "psfMatchingKernel_position_angle", "F",
            "PSF matching kernel centroid offset position angle.",
            units="radian")
        self.schema.addField(
            "psfMatchingKernel_direction", "F",
            "PSF matching kernel centroid offset direction in detector plane.",
            units="radian")

    @timeMethod
    def run(self, science, matchedTemplate, template, difference, diaSources, psfMatchingKernel):
        """Calculate difference image metrics on specific locations across the images

        Parameters
        ----------
        science : `lsst.afw.image.ExposureF`
            Science exposure that the template was subtracted from.
        matchedTemplate : `lsst.afw.image.ExposureF`
            Warped and PSF-matched template that was used produce the
            difference image.
        template : `lsst.afw.image.ExposureF`
            Warped and non PSF-matched template that was used produce
            the difference image.
        difference : `lsst.afw.image.ExposureF`
            Result of subtracting template from the science image.
        diaSources : `lsst.afw.table.SourceCatalog`
                The catalog of detected sources.
        psfMatchingKernel : `~lsst.afw.math.LinearCombinationKernel`
            The PSF matching kernel of the subtraction to evaluate.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            ``spatiallySampledMetrics`` :  `astropy.table.Table`
                Image quality metrics spatially sampled locations.
        """

        idFactory = lsst.meas.base.IdGenerator().make_table_id_factory()

        spatiallySampledMetrics = afwTable.SourceCatalog(self.schema)
        spatiallySampledMetrics.getTable().setIdFactory(idFactory)

        self.metricSources.run(mask=science.mask, seed=difference.info.id, catalog=spatiallySampledMetrics)

        metricsMaskPlanes = []
        for maskPlane in self.config.metricsMaskPlanes:
            try:
                metricsMaskPlanes.append(maskPlane)
            except InvalidParameterError:
                self.log.info("Unable to calculate metrics for mask plane %s: not in image"%maskPlane)

        for src in spatiallySampledMetrics:
            self._evaluateLocalMetric(src, science, matchedTemplate, template, difference, diaSources,
                                      metricsMaskPlanes=metricsMaskPlanes,
                                      psfMatchingKernel=psfMatchingKernel)

        return pipeBase.Struct(spatiallySampledMetrics=spatiallySampledMetrics.asAstropy())

    def _evaluateLocalMetric(self, src, science, matchedTemplate, template, difference, diaSources,
                             metricsMaskPlanes, psfMatchingKernel):
        """Calculate image quality metrics at spatially sampled locations.

        Parameters
        ----------
        src : `lsst.afw.table.SourceRecord`
            The source record to be updated with metric calculations.
        diaSources : `lsst.afw.table.SourceCatalog`
            The catalog of detected sources.
        science : `lsst.afw.image.Exposure`
            The science image.
        matchedTemplate : `lsst.afw.image.Exposure`
            The reference image, warped and psf-matched to the science image.
        difference : `lsst.afw.image.Exposure`
            Result of subtracting template from the science image.
        metricsMaskPlanes : `list` of `str`
            Mask planes to calculate metrics from.
        psfMatchingKernel : `~lsst.afw.math.LinearCombinationKernel`
            The PSF matching kernel of the subtraction to evaluate.
        """
        bbox = src.getFootprint().getBBox()
        pix = bbox.getCenter()
        src.set('science_psfSize', getPsfFwhm(science.psf, position=pix))
        try:
            src.set('template_psfSize', getPsfFwhm(template.psf, position=pix))
        except (InvalidParameterError, RangeError):
            src.set('template_psfSize', np.nan)

        metricRegionSize = 100
        bbox.grow(metricRegionSize)
        bbox = bbox.clippedTo(science.getBBox())
        nPix = bbox.getArea()
        pixScale = science.wcs.getPixelScale(bbox.getCenter())
        area = nPix*pixScale.asDegrees()**2
        peak = src.getFootprint().getPeaks()[0]
        src.set('x', peak['i_x'])
        src.set('y', peak['i_y'])
        src.setCoord(science.wcs.pixelToSky(peak['i_x'], peak['i_y']))
        selectSources = diaSources[bbox.contains(diaSources.getX(), diaSources.getY())]
        sourceDensity = len(selectSources)/area
        dipoleSources = selectSources[selectSources["ip_diffim_DipoleFit_flag_classification"]]
        dipoleDensity = len(dipoleSources)/area

        if dipoleSources:
            meanDipoleOrientation = angleMean(dipoleSources["ip_diffim_DipoleFit_orientation"])
            src.set('dipole_direction', meanDipoleOrientation)
            meanDipoleSeparation = np.mean(dipoleSources["ip_diffim_DipoleFit_separation"])
            src.set('dipole_separation', meanDipoleSeparation)

        templateVal = np.median(matchedTemplate[bbox].image.array)
        scienceVal = np.median(science[bbox].image.array)
        diffimVal = np.median(difference[bbox].image.array)
        src.set('source_density', sourceDensity)
        src.set('dipole_density', dipoleDensity)
        src.set('template_value', templateVal)
        src.set('science_value', scienceVal)
        src.set('diffim_value', diffimVal)
        for maskPlane in metricsMaskPlanes:
            src.set("%s_mask_fraction"%maskPlane.lower(),
                    evaluateMaskFraction(difference.mask[bbox], maskPlane)
                    )

        krnlSum, dx, dy, direction, length = getKernelCenterDisplacement(
            psfMatchingKernel, src.get('x'), src.get('y'))

        point1 = lsst.geom.SpherePoint(
            src.get('coord_ra'), src.get('coord_dec'),
            lsst.geom.radians)
        point2 = science.wcs.pixelToSky(src.get('x') + dx, src.get('y') + dy)
        bearing = point1.bearingTo(point2)
        pa_ref_angle = lsst.geom.Angle(np.pi/2, lsst.geom.radians)
        pa = pa_ref_angle - bearing
        # Wrap around to get Delta_RA from -pi to +pi
        pa = pa.wrapCtr()
        position_angle = pa.asRadians()

        src.set('psfMatchingKernel_sum', krnlSum)
        src.set('psfMatchingKernel_dx', dx)
        src.set('psfMatchingKernel_dy', dy)
        src.set('psfMatchingKernel_length', length*pixScale.asArcseconds())
        src.set('psfMatchingKernel_position_angle', position_angle)  # in E of N position angle
        src.set('psfMatchingKernel_direction', direction)  # direction offset in detector
