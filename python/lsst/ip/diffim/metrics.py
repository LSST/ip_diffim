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

from lsst.afw.table import SourceTable
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, PipelineTaskConnections, connectionTypes
from lsst.verify import Measurement
from lsst.verify.gen2tasks import register
from lsst.verify.tasks import MetricTask, MetricConfig, MetricConnections, \
    MetricComputationError


class _FakesAwareMetricConnections(PipelineTaskConnections, dimensions=set()):
    """A mixin for task connections that need to filter out fake sources.

    Requires that the associated config be a `_FakesAwareMetricConfig`.
    """
    matchedFakes = connectionTypes.Input(
        doc="A matched catalog of fakes and pipeline sources. The schema must "
            "include a column for source IDs.",
        # Do not use templates, to avoid injecting them into other classes
        name="fakes_deepDiff_matchDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )

    def __init__(self, *, config=None):
        """Optional input handling for `matchedFakes`.

        Parameters
        ----------
        config : `_FakesAwareMetricConfig`
            A config specifying, among other information, whether fakes should
            be handled.
        """
        super().__init__(config=config)

        if not config.removeFakes:
            self.inputs.remove("matchedFakes")


class _FakesAwareMetricConfig(Config):
    """A mixin for task configs that need to filter out fake sources.

    See also
    --------
    _FakesAwareMetricConnections
    """
    removeFakes = Field(
        dtype=bool,
        doc="If set, filter out any fake sources using a source match catalog.",
        default=False,
    )
    fakesSourceIdColumn = Field(
        doc="Column in matchedFakes containing the source ID.",
        dtype=str,
        default="diaSourceId",
    )


class NumberSciSourcesMetricConnections(
        MetricConnections,
        _FakesAwareMetricConnections,
        defaultTemplates={"package": "ip_diffim",
                          "metric": "numSciSources"},
        dimensions={"instrument", "visit", "detector"},
):
    sources = connectionTypes.Input(
        doc="The catalog of science sources.",
        name="src",
        storageClass="SourceCatalog",
        dimensions={"instrument", "visit", "detector"},
    )


class NumberSciSourcesMetricConfig(
        MetricConfig,
        _FakesAwareMetricConfig,
        pipelineConnections=NumberSciSourcesMetricConnections):
    pass


@register("numSciSources")
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

    def run(self, sources, matchedFakes=None):
        """Count the number of non-primary science sources.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog` or `None`
            A science source catalog, which may be empty or `None`.
       matchedFakes : `pandas.DataFrame` or `None`
            An optional catalog of sources to exclude as fakes.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A `~lsst.pipe.base.Struct` containing the following component:

            ``measurement``
                the total number of non-primary science sources
                (`lsst.verify.Measurement` or `None`)
        """
        if sources is not None:
            fakeIds = matchedFakes[self.config.fakesSourceIdColumn] if self.config.removeFakes else None
            nSciSources = _countRealSources(sources, fakeIds)
            meas = Measurement(self.config.metricName, nSciSources * u.count)
        else:
            self.log.info("Nothing to do: no catalogs found.")
            meas = None
        return Struct(measurement=meas)


class FractionDiaSourcesToSciSourcesMetricConnections(
        MetricTask.ConfigClass.ConnectionsClass,
        _FakesAwareMetricConnections,
        dimensions={"instrument", "visit", "detector"},
        defaultTemplates={"coaddName": "deep",
                          "fakesType": "",
                          "package": "ip_diffim",
                          "metric": "fracDiaSourcesToSciSources"}):
    sciSources = connectionTypes.Input(
        doc="The catalog of science sources.",
        name="src",
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
        _FakesAwareMetricConfig,
        pipelineConnections=FractionDiaSourcesToSciSourcesMetricConnections):
    pass


@register("fracDiaSourcesToSciSources")
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

    def run(self, sciSources, diaSources, matchedFakes=None):
        """Compute the ratio of DIASources to non-primary science sources.

        Parameters
        ----------
        sciSources : `lsst.afw.table.SourceCatalog` or `None`
            A science source catalog, which may be empty or `None`.
        diaSources : `lsst.afw.table.SourceCatalog` or `None`
            A DIASource catalog for the same unit of processing
            as ``sciSources``.
       matchedFakes : `pandas.DataFrame` or `None`
            An optional catalog of sources to exclude as fakes.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A `~lsst.pipe.base.Struct` containing the following component:

            ``measurement``
                the ratio (`lsst.verify.Measurement` or `None`)
        """
        if diaSources is not None and sciSources is not None:
            fakeIds = matchedFakes[self.config.fakesSourceIdColumn] if self.config.removeFakes else None
            nSciSources = _countRealSources(sciSources, fakeIds)
            nDiaSources = _countRealSources(diaSources, fakeIds)
            metricName = self.config.metricName
            if nSciSources <= 0:
                raise MetricComputationError(
                    "No science sources found; ratio of DIASources to science sources ill-defined.")
            else:
                meas = Measurement(metricName, nDiaSources / nSciSources * u.dimensionless_unscaled)
        else:
            self.log.info("Nothing to do: no catalogs found.")
            meas = None
        return Struct(measurement=meas)


def _countRealSources(catalog, fakes):
    """Return the number of valid sources in a catalog.

    At present, this definition includes only primary sources. If a catalog
    does not have a ``detect_isPrimary`` flag, this function counts non-sky
    sources. If it does not have a ``sky_source`` flag, either, all sources
    are counted.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        The catalog of sources to count.
    fakes : `numpy.ndarray` [`int`], (Nf,) or `None`
        An optional array listing the IDs of any sources matched to fakes.
        These do not, in general, have either the same length or the same order
        as ``catalog``. Some entries may be 0 to indicate an unmatched fake.

    Returns
    -------
    count : `int`
        The number of sources that satisfy the criteria.
    """
    if fakes is not None:
        nonFake = np.in1d(catalog[SourceTable.getIdKey()], fakes, invert=True)
    else:
        nonFake = np.full_like(catalog[SourceTable.getIdKey()], True, dtype=bool)

    # E712 is not applicable, because afw.table.SourceRecord.ColumnView
    # is not a bool.
    if "detect_isPrimary" in catalog.schema:
        return np.count_nonzero(nonFake & (catalog["detect_isPrimary"] == True))  # noqa: E712
    elif "sky_source" in catalog.schema:
        return np.count_nonzero(nonFake & (catalog["sky_source"] == False))  # noqa: E712
    else:
        return np.count_nonzero(nonFake)
