# This file is part of ip_diffim.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__all__ = [
    "NumberSciSourcesMetricTask", "NumberSciSourcesMetricConfig",
    "FractionDiaSourcesToSciSourcesMetricTask", "FractionDiaSourcesToSciSourcesMetricConfig",
]


import numpy as np
import astropy.units as u

from lsst.pipe.base import Struct, connectionTypes
from lsst.verify import Measurement
from lsst.verify.tasks import MetricTask, MetricConfig, MetricConnections, \
    MetricComputationError


class NumberSciSourcesMetricConnections(
        MetricConnections,
        defaultTemplates={"package": "ip_diffim",
                          "metric": "numSciSources"},
        dimensions={"instrument", "visit", "detector"},
):
    sources = connectionTypes.Input(
        doc="The catalog of science sources.",
        name="initial_stars_footprints_detector",
        storageClass="SourceCatalog",
        dimensions={"instrument", "visit", "detector"},
    )


class NumberSciSourcesMetricConfig(
        MetricConfig,
        pipelineConnections=NumberSciSourcesMetricConnections):
    pass


class NumberSciSourcesMetricTask(MetricTask):
    """Task that computes the number of cataloged non-primary science sources.

    Notes
    -----
    The task excludes any non-primary sources in the catalog, but it does
    not require that the catalog include a ``detect_isPrimary`` or
    ``sky_sources`` column.
    """
    _DefaultName = "numSciSources"
    ConfigClass = NumberSciSourcesMetricConfig

    def run(self, sources):
        """Count the number of primary science sources.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog`
            A science source catalog, which may be empty.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A `~lsst.pipe.base.Struct` containing the following component:

            ``measurement``
                the total number of non-primary science sources
                (`lsst.verify.Measurement` or `None`)
        """
        nSciSources = _countRealSources(sources)
        meas = Measurement(self.config.metricName, nSciSources * u.count)
        return Struct(measurement=meas)


class FractionDiaSourcesToSciSourcesMetricConnections(
        MetricTask.ConfigClass.ConnectionsClass,
        dimensions={"instrument", "visit", "detector"},
        defaultTemplates={"coaddName": "deep",
                          "fakesType": "",
                          "package": "ip_diffim",
                          "metric": "fracDiaSourcesToSciSources"}):
    sciSources = connectionTypes.Input(
        doc="The catalog of science sources.",
        name="initial_stars_footprints_detector",
        storageClass="SourceCatalog",
        dimensions={"instrument", "visit", "detector"},
    )
    diaSources = connectionTypes.Input(
        doc="The catalog of DIASources.",
        name="{fakesType}{coaddName}Diff_diaSrc",
        storageClass="SourceCatalog",
        dimensions={"instrument", "visit", "detector"},
    )


class FractionDiaSourcesToSciSourcesMetricConfig(
        MetricTask.ConfigClass,
        pipelineConnections=FractionDiaSourcesToSciSourcesMetricConnections):
    pass


class FractionDiaSourcesToSciSourcesMetricTask(MetricTask):
    """Task that computes the ratio of difference image sources to science
    sources in an image, visit, etc.

    Notes
    -----
    The task excludes any non-primary sources in the catalog, but it does
    not require that the catalog include a ``detect_isPrimary`` or
    ``sky_sources`` column.
    """
    _DefaultName = "fracDiaSourcesToSciSources"
    ConfigClass = FractionDiaSourcesToSciSourcesMetricConfig

    def run(self, sciSources, diaSources):
        """Compute the ratio of DIASources to non-primary science sources.

        Parameters
        ----------
        sciSources : `lsst.afw.table.SourceCatalog`
            A science source catalog, which may be empty.
        diaSources : `lsst.afw.table.SourceCatalog`
            A DIASource catalog for the same unit of processing
            as ``sciSources``.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A `~lsst.pipe.base.Struct` containing the following component:

            ``measurement``
                the ratio (`lsst.verify.Measurement` or `None`)
        """
        nSciSources = _countRealSources(sciSources)
        nDiaSources = _countRealSources(diaSources)
        metricName = self.config.metricName
        if nSciSources <= 0:
            raise MetricComputationError(
                "No science sources found; ratio of DIASources to science sources ill-defined.")
        else:
            meas = Measurement(metricName, nDiaSources / nSciSources * u.dimensionless_unscaled)
        return Struct(measurement=meas)


def _countRealSources(catalog):
    """Return the number of valid sources in a catalog.

    At present, this definition includes only primary sources. If a catalog
    does not have a ``detect_isPrimary`` flag, this function counts non-sky
    sources. If it does not have a ``sky_source`` flag, either, all sources
    are counted.

    Parameters
    ----------
    `catalog` : `lsst.afw.table.SourceCatalog`
        The catalog of sources to count.

    Returns
    -------
    count : `int`
        The number of sources that satisfy the criteria.
    """
    # E712 is not applicable, because afw.table.SourceRecord.ColumnView
    # is not a bool.
    if "detect_isPrimary" in catalog.schema:
        return np.count_nonzero(catalog["detect_isPrimary"] == True)  # noqa: E712
    elif "sky_source" in catalog.schema:
        return np.count_nonzero(catalog["sky_source"] == False)  # noqa: E712
    else:
        return len(catalog)
