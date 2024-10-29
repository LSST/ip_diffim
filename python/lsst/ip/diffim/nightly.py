import numpy as np
import pandas as pd
import astropy.units as u

import lsst.afw.table as afwTable
import lsst.daf.base as dafBase

# import lsst.meas.extensions.trailedSources  # noqa: F401
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.geom as geom
import lsst.utils
import lsst.pipe.base.connectionTypes as connTypes

from lsst.daf.base import DateTime
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.base import ForcedMeasurementTask, ApplyApCorrTask
from lsst.utils.timer import timeMethod
from lsst.skymap import BaseSkyMap
from lsst.pipe.tasks.coaddBase import makeSkyInfo

# from lsst.pipe.tasks.functors import Column
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks.simpleAssociation import (
    SimpleAssociationConfig,
    SimpleAssociationTask,
)
from astropy.time import Time
from lsst.meas.base import IdGenerator
from lsst.pipe.tasks.associationUtils import query_disc, eq2xyz, toIndex

from lsst.drp.tasks.assemble_coadd import (
    AssembleCoaddConnections,
    AssembleCoaddConfig,
    AssembleCoaddTask,
)
from lsst.pipe.tasks.postprocess import (
    TransformCatalogBaseTask,
    TransformCatalogBaseConfig,
)
from lsst.ap.association.transformDiaSourceCatalog import (
    TransformDiaSourceCatalogConfig,
    TransformDiaSourceCatalogTask,
    getSignificance,
)
from lsst.pipe.tasks.drpAssociationPipe import (
    DrpAssociationPipeConfig,
    DrpAssociationPipeTask,
)

from lsst.pipe.tasks.drpDiaCalculationPipe import (
    DrpDiaCalculationPipeConfig,
    DrpDiaCalculationPipeTask,
)

from . import DipoleFitTask
from . import DetectAndMeasureConfig, DetectAndMeasureTask
from . import AlardLuptonSubtractConfig, AlardLuptonSubtractTask

__all__ = [
    "SimpleAssociationCoaddConfig",
    "SimpleAssociationCoaddTask",
    "AssembleNightlyCoaddConfig",
    "AssembleNightlyCoaddTask",
    "DetectCoaddForDiffConfig",
    "DetectCoaddForDiffTask",
    "CoaddAlardLuptonSubtractConfig",
    "CoaddAlardLuptonSubtractTask",
    "DetectAndMeasureCoaddConfig",
    "DetectAndMeasureCoaddTask",
    "TransformCoaddDiaSourceCatalogConfig",
    "TransformCoaddDiaSourceCatalogTask",
    "DrpCoaddAssociationPipeConfig",
    "DrpCoaddAssociationPipeTask",
    "ForcedPhotCoaddFromDataFrameConfig",
    "ForcedPhotCoaddFromDataFrameTask",
]


##############################################################################################################
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base import (
    Struct,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
)
from lsst.pex.config import Field, ConfigurableField, ChoiceField
from lsst.meas.algorithms import (
    SubtractBackgroundTask,
    SourceDetectionTask,
    MeasureApCorrTask,
    MeasureApCorrError,
)
from lsst.meas.base import (
    SingleFrameMeasurementTask,
    ApplyApCorrTask,
    CatalogCalculationTask,
    SkyMapIdGeneratorConfig,
)
from lsst.daf.butler import (
    DimensionPacker,
    DimensionGraph,
    DimensionGroup,
    DataCoordinate,
)
from lsst.skymap.packers import SkyMapDimensionPackerConfig, SkyMapDimensionPacker
from collections.abc import Mapping


class SkyMapNightDimensionPackerConfig(SkyMapDimensionPackerConfig):
    n_nights = Field(
        "Number of nights to be packed",
        dtype=int,
        optional=True,
        default=None,
    )


class SkyMapNightDimensionPacker(SkyMapDimensionPacker):
    """A `DimensionPacker` for tract, patch, night and optionally band,
    given a SkyMap.
    
    Calculates nights from Jan, 1, 2020 and assumes no more than 5000 nights
    from this date. These are currently hardcoded until I can figure out how
    to specify them in some configuration.

    Parameters
    ----------
    fixed : `lsst.daf.butler.DataCoordinate`
        Data ID that identifies just the ``skymap`` dimension.  Must have
        dimension records attached unless ``n_tracts`` and ``n_patches`` are
        not `None`.
    dimensions : `lsst.daf.butler.DimensionGraph`, or \
            `lsst.daf.butler.DimensionGroup`, optional
        The dimensions of data IDs packed by this instance.  Must include
        ``{skymap, tract, patch}``, and may include ``band``.  If not provided,
        this will be set to include ``band`` if ``n_bands != 0``.
    bands : `~collections.abc.Mapping` [ `str`, `int` ] or `None`, optional
        Mapping from band name to integer to use in the packed ID.  `None` uses
        a fixed set of bands defined in this class.  When calling code can
        enumerate the bands it is likely to see, providing an explicit mapping
        is preferable.
    n_bands : `int` or `None`, optional
        The number of bands to leave room for in the packed ID.  If `None`,
        this will be set to ``len(bands)``.  If ``0``, the band will not be
        included in the dimensions at all.  If ``1``, the band will be included
        in the dimensions but will not occupy any extra bits in the packed ID.
        This may be larger or smaller than ``len(bands)``, to reserve extra
        space for the future or align to byte boundaries, or support a subset
        of a larger mapping, respectively.
    n_tracts : `int` or `None`, optional
        The number of tracts to leave room for in the packed ID.  If `None`,
        this will be set via the ``skymap`` dimension record in ``fixed``.
    n_patches : `int` or `None`, optional
        The number of patches (per tract) to leave room for in the packed ID.
        If `None`, this will be set via the ``skymap`` dimension record in
        ``fixed``.

    Notes
    -----
    The standard pattern for constructing instances of this class is to use
    `make_config_field`::

        class SomeConfig(lsst.pex.config.Config):
            packer = ObservationDimensionPacker.make_config_field()

        class SomeTask(lsst.pipe.base.Task):
            ConfigClass = SomeConfig

            def run(self, ..., data_id: DataCoordinate):
                packer = self.config.packer.apply(data_id)
                packed_id = packer.pack(data_id)
                ...

    """

    ConfigClass = SkyMapNightDimensionPackerConfig

    def __init__(
        self,
        fixed: DataCoordinate,
        day_zero: Time = Time("2020-01-01"),
        dimensions: DimensionGroup | DimensionGraph | None = None,
        bands: Mapping[str, int] | None = None,
        n_bands: int | None = None,
        n_tracts: int | None = None,
        n_patches: int | None = None,
        n_nights: int | None = None,
    ):
        # Copied logic from SkyMapDimensionPacker with addition of "day_obs" and "instrument"
        if bands is None:
            bands = {b: i for i, b in enumerate(self.SUPPORTED_FILTERS)}
        if dimensions is None:
            if n_bands is None:
                n_bands = len(bands)
            dimension_names = ["tract", "patch", "day_obs", "instrument"]
            if n_bands != 0:
                dimension_names.append("band")
            dimensions = fixed.universe.conform(dimension_names)
        else:
            if "band" not in dimensions.names:
                n_bands = 0
                if dimensions.names != {
                    "tract",
                    "patch",
                    "skymap",
                    "day_obs",
                    "instrument",
                }:
                    raise ValueError(
                        f"Invalid dimensions for skymap dimension packer with n_bands=0: {dimensions}."
                    )
            else:
                if dimensions.names != {
                    "tract",
                    "patch",
                    "skymap",
                    "band",
                    "day_obs",
                    "instrument",
                }:
                    raise ValueError(
                        f"Invalid dimensions for skymap dimension packer with n_bands>0: {dimensions}."
                    )
                if n_bands is None:
                    n_bands = len(bands)
        if n_tracts is None:
            n_tracts = fixed.records["skymap"].tract_max
        if n_patches is None:
            n_patches = (
                fixed.records["skymap"].patch_nx_max
                * fixed.records["skymap"].patch_ny_max
            )
        DimensionPacker.__init__(self, fixed, dimensions)

        if n_nights is None:
            n_nights = 5000

        self._bands = bands
        self._n_bands = n_bands
        self._n_tracts = n_tracts
        self._n_patches = n_patches
        self._bands_list = None
        self._n_nights = n_nights
        self.day_zero = day_zero

    @classmethod
    def make_config_field(
        cls, doc: str = "How to pack tract, patch, and possibly band into an integer."
    ) -> ConfigurableField:
        """Make a config field to control how skymap data IDs are packed.

        Parameters
        ----------
        doc : `str`, optional
            Documentation for the config field.

        Returns
        -------
        field : `lsst.pex.config.ConfigurableField`
            A config field whose instance values are [wrapper proxies to]
            `SkyMapDimensionPackerConfig` instances.
        """
        return ConfigurableField(
            doc, target=cls.from_config, ConfigClass=cls.ConfigClass
        )

    @classmethod
    def from_config(
        cls, data_id: DataCoordinate, config: SkyMapDimensionPackerConfig
    ) -> SkyMapDimensionPacker:
        """Construct a dimension packer from a config object and a data ID.

        Parameters
        ----------
        data_id : `lsst.daf.butler.DataCoordinate`
            Data ID that identifies at least the ``skymap`` dimension.  Must
            have dimension records attached unless ``config.n_tracts`` and
            ``config.n_patches`` are both not `None`.
        config : `SkyMapDimensionPackerConfig`
            Configuration object.

        Returns
        -------
        packer : `SkyMapDimensionPackerConfig`
            New dimension packer.

        Notes
        -----
        This interface is provided for consistency with the `lsst.pex.config`
        "Configurable" concept, and is invoked when ``apply(data_id)`` is
        called on a config instance attribute that corresponds to a field
        created by `make_config_field`.  The constructor signature cannot play
        this role easily for backwards compatibility reasons.
        """
        return cls(
            data_id.subset(data_id.universe.conform(["skymap", "instrument"])),
            n_bands=config.n_bands,
            bands=config.bands,
            n_tracts=config.n_tracts,
            n_patches=config.n_patches,
            n_nights=config.n_nights,
        )

    @property
    def maxBits(self) -> int:
        # Docstring inherited from DataIdPacker.maxBits
        packedMax = self._n_tracts * self._n_patches * self._n_nights
        if self._n_bands:
            packedMax *= self._n_bands
        return (packedMax - 1).bit_length()

    def _pack(self, dataId: DataCoordinate) -> int:
        if dataId["patch"] >= self._n_patches:
            raise ValueError(
                f"Patch ID {dataId['patch']} is out of bounds; expected <{self._n_patches}."
            )
        if dataId["tract"] >= self._n_tracts:
            raise ValueError(
                f"Tract ID {dataId['tract']} is out of bounds; expected <{self._n_tracts}."
            )
        day_obs_str = str(dataId["day_obs"])
        time_str = day_obs_str[:4] + "-" + day_obs_str[4:6] + "-" + day_obs_str[6:8]
        date = Time(time_str)
        day_n = (date - self.day_zero).datetime.days
        if day_n >= self._n_nights:
            raise ValueError(
                f"day_obs {dataId['day_obs']} is out of bounds; expected <{self.config.day_zero.value[:10]+self._n_nights}."
            )

        packed = dataId["patch"] + self._n_patches * dataId["tract"]
        packed += day_n * self._n_patches * self._n_tracts

        if self._n_bands:
            if (band_index := self._bands.get(dataId["band"])) is None:
                raise ValueError(
                    f"Band {dataId['band']!r} is not supported by SkyMapDimensionPacker "
                    f"configuration; expected one of {list(self._bands)}."
                )
            if band_index >= self._n_bands:
                raise ValueError(
                    f"Band index {band_index} for {dataId['band']!r} is out of bounds; "
                    f"expected <{self._n_bands}."
                )
            packed += (
                self._bands[dataId["band"]]
                * self._n_patches
                * self._n_tracts
                * self._n_nights
            )

        return packed

    def unpack(self, packedId: int) -> DataCoordinate:
        # Docstring inherited from DataIdPacker.unpack
        d = {"skymap": self.fixed["skymap"], "instrument": self.fixed["instrument"]}
        if self._n_bands:
            index, packedId = divmod(
                packedId, (self._n_tracts * self._n_patches * self._n_nights)
            )
            if self._bands_list is None:
                self._bands_list = list(self._bands)
            d["band"] = self._bands_list[index]

        index2, packedId2 = divmod(packedId, (self._n_tracts * self._n_patches))
        night = index2 * u.day + self.day_zero
        d["day_obs"] = int(night.value[:10].replace("-", ""))
        d["tract"], d["patch"] = divmod(packedId2, self._n_patches)
        return DataCoordinate.standardize(d, dimensions=self.dimensions)


class SkyMapNightIdGeneratorConfig(SkyMapIdGeneratorConfig):
    """Configuration class for generating integer IDs from
    ``{tract, patch, [band]}`` data IDs.

    See `IdGenerator` for usage.
    """

    packer = SkyMapNightDimensionPacker.make_config_field()

    def _make_dimension_packer(self, data_id: DataCoordinate) -> DimensionPacker:
        # Docstring inherited.
        return self.packer.apply(data_id)


class SimpleAssociationCoaddConfig(SimpleAssociationConfig):
    """Configuration parameters for the SimpleAssociationCoaddTask"""

    pass


class SimpleAssociationCoaddTask(pipeBase.Task):
    """Construct DiaObjects from a DataFrame of DIASources by spatially
    associating the sources.

    Represents a simple, brute force algorithm, 2-way matching of DiaSources
    into. DiaObjects. Algorithm picks the nearest, first match within the
    matching radius of a DiaObject to associate a source to for simplicity.
    """

    ConfigClass = SimpleAssociationCoaddConfig
    _DefaultName = "simpleAssociationCoadd"

    def run(self, diaSources, idGenerator=None):
        """Associate DiaSources into a collection of DiaObjects using a
        brute force matching algorithm.

        Reproducible for the same input data is assured by ordering the
        DiaSource data by visit,detector.

        Parameters
        ----------
        diaSources : `pandas.DataFrame`
            DiaSources in clusters of visit, detector to spatially associate
            into DiaObjects.
        idGenerator : `lsst.meas.base.IdGenerator`, optional
            Object that generates Object IDs and random number generator seeds.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``assocDiaSources``
                Table of DiaSources with updated values for the DiaObjects
                they are spatially associated to (`pandas.DataFrame`).
            ``diaObjects``
                Table of DiaObjects from matching DiaSources
                (`pandas.DataFrame`).
        """
        # Expected indexes include diaSourceId or meaningless range index
        # If meaningless range index, drop it, else keep it.
        doDropIndex = diaSources.index.names[0] is None
        diaSources.reset_index(inplace=True, drop=doDropIndex)

        # Sort by skyId and diaSourceId to get a reproducible
        # ordering for the association.
        diaSources.set_index(["skyId", "diaSourceId"], inplace=True)
        diaSources = diaSources.sort_index()

        # Empty lists to store matching and location data.
        diaObjectCat = []
        diaObjectCoords = []
        healPixIndices = []

        # Create Id factory and catalog for creating DiaObjectIds.
        if idGenerator is None:
            idGenerator = IdGenerator()
        idCat = idGenerator.make_source_catalog(
            afwTable.SourceTable.makeMinimalSchema()
        )

        for skyId in diaSources.index.levels[0]:
            # For the first visit,detector, just copy the DiaSource info into the
            # diaObject data to create the first set of Objects.
            orderedSources = diaSources.loc[skyId]
            if len(diaObjectCat) == 0:
                for diaSourceId, diaSrc in orderedSources.iterrows():
                    self.addNewDiaObject(
                        diaSrc,
                        diaSources,
                        skyId,
                        diaSourceId,
                        diaObjectCat,
                        idCat,
                        diaObjectCoords,
                        healPixIndices,
                    )
                continue
            # Temp list to store DiaObjects already used for this visit, detector.
            usedMatchIndicies = []
            # Run over subsequent data.
            for diaSourceId, diaSrc in orderedSources.iterrows():
                # Find matches.
                matchResult = self.findMatches(
                    diaSrc["ra"],
                    diaSrc["dec"],
                    2 * self.config.tolerance,
                    healPixIndices,
                    diaObjectCat,
                )
                dists = matchResult.dists
                matches = matchResult.matches
                # Create a new DiaObject if no match found.
                if dists is None:
                    self.addNewDiaObject(
                        diaSrc,
                        diaSources,
                        skyId,
                        diaSourceId,
                        diaObjectCat,
                        idCat,
                        diaObjectCoords,
                        healPixIndices,
                    )
                    continue
                # If matched, update catalogs and arrays.
                if np.min(dists) < np.deg2rad(self.config.tolerance / 3600):
                    matchDistArg = np.argmin(dists)
                    matchIndex = matches[matchDistArg]
                    # Test to see if the DiaObject has been used.
                    if np.isin([matchIndex], usedMatchIndicies).sum() < 1:
                        self.updateCatalogs(
                            matchIndex,
                            diaSrc,
                            diaSources,
                            skyId,
                            diaSourceId,
                            diaObjectCat,
                            diaObjectCoords,
                            healPixIndices,
                        )
                        usedMatchIndicies.append(matchIndex)
                    # If the matched DiaObject has already been used, create a
                    # new DiaObject for this DiaSource.
                    else:
                        self.addNewDiaObject(
                            diaSrc,
                            diaSources,
                            skyId,
                            diaSourceId,
                            diaObjectCat,
                            idCat,
                            diaObjectCoords,
                            healPixIndices,
                        )
                # Create new DiaObject if no match found within the matching
                # tolerance.
                else:
                    self.addNewDiaObject(
                        diaSrc,
                        diaSources,
                        skyId,
                        diaSourceId,
                        diaObjectCat,
                        idCat,
                        diaObjectCoords,
                        healPixIndices,
                    )

        # Drop indices before returning associated diaSource catalog.
        diaSources.reset_index(inplace=True)
        diaSources.set_index("diaSourceId", inplace=True, verify_integrity=True)

        objs = (
            diaObjectCat
            if diaObjectCat
            else np.array(
                [],
                dtype=[
                    ("diaObjectId", "int64"),
                    ("ra", "float64"),
                    ("dec", "float64"),
                    ("nDiaSources", "int64"),
                ],
            )
        )
        diaObjects = pd.DataFrame(data=objs)

        if "diaObjectId" in diaObjects.columns:
            diaObjects.set_index("diaObjectId", inplace=True, verify_integrity=True)

        return pipeBase.Struct(assocDiaSources=diaSources, diaObjects=diaObjects)

    def addNewDiaObject(
        self,
        diaSrc,
        diaSources,
        skyId,
        diaSourceId,
        diaObjCat,
        idCat,
        diaObjCoords,
        healPixIndices,
    ):
        """Create a new DiaObject and append its data.

         Parameters
         ----------
         diaSrc : `pandas.Series`
             Full unassociated DiaSource to create a DiaObject from.
         diaSources : `pandas.DataFrame`
             DiaSource catalog to update information in. The catalog is
             modified in place. Must be indexed on:
             `(visit, detector), diaSourceId`.
         skyId : `int`
             Sky ids where ``diaSrc`` was observed.
        diaSourceId : `int`
             Unique identifier of the DiaSource.
         diaObjectCat : `list` of `dict`s
             Catalog of diaObjects to append the new object o.
         idCat : `lsst.afw.table.SourceCatalog`
             Catalog with the IdFactory used to generate unique DiaObject
             identifiers.
         diaObjectCoords : `list` of `list`s of `lsst.geom.SpherePoint`s
             Set of coordinates of DiaSource locations that make up the
             DiaObject average coordinate.
         healPixIndices : `list` of `int`s
             HealPix indices representing the locations of each currently
             existing DiaObject.
        """
        hpIndex = toIndex(self.config.nside, diaSrc["ra"], diaSrc["dec"])
        healPixIndices.append(hpIndex)

        sphPoint = geom.SpherePoint(diaSrc["ra"], diaSrc["dec"], geom.degrees)
        diaObjCoords.append([sphPoint])

        diaObjId = idCat.addNew().get("id")
        diaObjCat.append(self.createDiaObject(diaObjId, diaSrc["ra"], diaSrc["dec"]))
        diaSources.loc[(skyId, diaSourceId), "diaObjectId"] = diaObjId

    def updateCatalogs(
        self,
        matchIndex,
        diaSrc,
        diaSources,
        skyId,
        diaSourceId,
        diaObjCat,
        diaObjCoords,
        healPixIndices,
    ):
        """Update DiaObject and DiaSource values after an association.

        Parameters
        ----------
        matchIndex : `int`
            Array index location of the DiaObject that ``diaSrc`` was
            associated to.
        diaSrc : `pandas.Series`
            Full unassociated DiaSource to create a DiaObject from.
        diaSources : `pandas.DataFrame`
            DiaSource catalog to update information in. The catalog is
            modified in place. Must be indexed on:
            `(visit, detector), diaSourceId`.
        visit, detector : `int`
            Visit and detector ids where ``diaSrc`` was observed.
        diaSourceId : `int`
            Unique identifier of the DiaSource.
        diaObjectCat : `list` of `dict`s
            Catalog of diaObjects to append the new object o.
        diaObjectCoords : `list` of `list`s of `lsst.geom.SpherePoint`s
            Set of coordinates of DiaSource locations that make up the
            DiaObject average coordinate.
        healPixIndices : `list` of `int`s
            HealPix indices representing the locations of each currently
            existing DiaObject.
        """
        # Update location and healPix index.
        sphPoint = geom.SpherePoint(diaSrc["ra"], diaSrc["dec"], geom.degrees)
        diaObjCoords[matchIndex].append(sphPoint)
        aveCoord = geom.averageSpherePoint(diaObjCoords[matchIndex])
        diaObjCat[matchIndex]["ra"] = aveCoord.getRa().asDegrees()
        diaObjCat[matchIndex]["dec"] = aveCoord.getDec().asDegrees()
        nSources = diaObjCat[matchIndex]["nDiaSources"]
        diaObjCat[matchIndex]["nDiaSources"] = nSources + 1
        healPixIndices[matchIndex] = toIndex(
            self.config.nside, diaObjCat[matchIndex]["ra"], diaObjCat[matchIndex]["dec"]
        )
        # Update DiaObject Id that this source is now associated to.
        diaSources.loc[(skyId, diaSourceId), "diaObjectId"] = diaObjCat[matchIndex][
            "diaObjectId"
        ]

    def findMatches(self, src_ra, src_dec, tol, hpIndices, diaObjs):
        """Search healPixels around DiaSource locations for DiaObjects.

        Parameters
        ----------
        src_ra : `float`
            DiaSource RA location.
        src_dec : `float`
            DiaSource Dec location.
        tol : `float`
            Size of annulus to convert to covering healPixels and search for
            DiaObjects.
        hpIndices : `list` of `int`s
            List of heal pix indices containing the DiaObjects in ``diaObjs``.
        diaObjs : `list` of `dict`s
            Catalog diaObjects to with full location information for comparing
            to DiaSources.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct containing

            ``dists``
                Array of distances between the current DiaSource diaObjects.
                (`numpy.ndarray` or `None`)
            ``matches``
                Array of array indices of diaObjects this DiaSource matches to.
                (`numpy.ndarray` or `None`)
        """
        match_indices = query_disc(
            self.config.nside, src_ra, src_dec, np.deg2rad(tol / 3600.0)
        )
        matchIndices = np.argwhere(np.isin(hpIndices, match_indices)).flatten()

        if len(matchIndices) < 1:
            return pipeBase.Struct(dists=None, matches=None)

        dists = np.array(
            [
                np.sqrt(
                    np.sum(
                        (
                            eq2xyz(src_ra, src_dec)
                            - eq2xyz(diaObjs[match]["ra"], diaObjs[match]["dec"])
                        )
                        ** 2
                    )
                )
                for match in matchIndices
            ]
        )
        return pipeBase.Struct(dists=dists, matches=matchIndices)

    def createDiaObject(self, objId, ra, dec):
        """Create a simple empty DiaObject with location and id information.

        Parameters
        ----------
        objId : `int`
            Unique ID for this new DiaObject.
        ra : `float`
            RA location of this DiaObject.
        dec : `float`
            Dec location of this DiaObject

        Returns
        -------
        DiaObject : `dict`
            Dictionary of values representing a DiaObject.
        """
        new_dia_object = {"diaObjectId": objId, "ra": ra, "dec": dec, "nDiaSources": 1}
        return new_dia_object


class AssembleNightlyCoaddConnections(
    AssembleCoaddConnections,
    dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "deepNightly",
        "warpType": "direct",
        "warpTypeSuffix": "",
    },
):
    inputWarps = pipeBase.connectionTypes.Input(
        doc=(
            "Input list of warps to be assemebled i.e. stacked."
            "WarpType (e.g. direct, psfMatched) is controlled by the "
            "warpType config parameter"
        ),
        name="{inputCoaddName}Coadd_{warpType}Warp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )
    selectedVisits = pipeBase.connectionTypes.Input(
        doc="Selected visits to be coadded.",
        name="{outputCoaddName}Visits",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "tract", "patch", "skymap", "band", "day_obs"),
    )
    coaddExposure = pipeBase.connectionTypes.Output(
        doc="Output coadded exposure, produced by stacking input warps",
        name="{outputCoaddName}Coadd{warpTypeSuffix}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band", "day_obs"),
    )
    nImage = pipeBase.connectionTypes.Output(
        doc="Output image of number of input images per pixel",
        name="{outputCoaddName}Coadd_nImage",
        storageClass="ImageU",
        dimensions=("tract", "patch", "skymap", "band", "day_obs"),
    )
    inputMap = pipeBase.connectionTypes.Output(
        doc="Output healsparse map of input images",
        name="{outputCoaddName}Coadd_inputMap",
        storageClass="HealSparseMap",
        dimensions=("tract", "patch", "skymap", "band", "day_obs"),
    )


class AssembleNightlyCoaddConfig(
    AssembleCoaddConfig, pipelineConnections=AssembleNightlyCoaddConnections
):
    def setDefaults(self):
        super().setDefaults()
        self.doInputMap = True


class AssembleNightlyCoaddTask(AssembleCoaddTask):
    """Assemble a coadd image from a set of warps from the same night

    This should have all the functionality of the `AssembleCoaddTask`
    with the additional dimensions of day_obs to enable running on
    different nights.

    """

    ConfigClass = AssembleNightlyCoaddConfig
    _DefaultName = "assembleNightlyCoadd"


class DetectCoaddForDiffConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    defaultTemplates={
        "inputCoaddName": "deepNightly",
        "outputCoaddName": "deepNightly",
    },
):
    detectionSchema = cT.InitOutput(
        doc="Schema of the detection catalog",
        name="{outputCoaddName}Coadd_det_schema",
        storageClass="SourceCatalog",
    )
    exposure = cT.Input(
        doc="Exposure on which detections are to be performed",
        name="{inputCoaddName}Coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    )
    outputSources = cT.Output(
        doc="Detected sources catalog",
        name="{outputCoaddName}Coadd_det",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    )
    outputExposure = cT.Output(
        doc="Exposure post detection",
        name="{outputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    )


class DetectCoaddForDiffConfig(
    PipelineTaskConfig, pipelineConnections=DetectCoaddForDiffConnections
):
    """Configuration parameters for the DetectCoaddSourcesTask"""

    doScaleVariance = Field(
        dtype=bool, default=True, doc="Scale variance plane using empirical noise?"
    )
    detection = ConfigurableField(target=SourceDetectionTask, doc="Source detection")
    coaddName = Field(dtype=str, default="deepNightly", doc="Name of coadd")
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask, doc="Measure sources"
    )
    idGenerator = SkyMapNightIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        self.detection.thresholdType = "pixel_stdev"
        self.detection.isotropicGrow = True
        # Coadds are made from background-subtracted CCDs, so any background subtraction should be very basic
        self.detection.reEstimateBackground = False
        self.detection.background.useApprox = False
        self.detection.background.binSize = 4096
        self.detection.background.undersampleStyle = "REDUCE_INTERP_ORDER"
        self.detection.doTempWideBackground = (
            True  # Suppress large footprints that overwhelm the deblender
        )
        # Include band in packed data IDs that go into object IDs (None -> "as
        # many bands as are defined", rather than the default of zero).
        self.idGenerator.packer.n_bands = None

        self.measurement.plugins.names = [
            "base_PixelFlags",
            "base_SdssCentroid",
            "ext_shapeHSM_HsmSourceMoments",
            "base_GaussianFlux",
            "base_PsfFlux",
            "base_CircularApertureFlux",
            "base_SkyCoord",
        ]
        self.measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"


class DetectCoaddForDiffTask(PipelineTask):
    """Detect sources on a single filter coadd.

    Coadding individual visits requires each exposure to be warped. This
    introduces covariance in the noise properties across pixels. Before
    detection, we correct the coadd variance by scaling the variance plane in
    the coadd to match the observed variance. This is an approximate
    approach -- strictly, we should propagate the full covariance matrix --
    but it is simple and works well in practice.

    After scaling the variance plane, we detect sources and generate footprints
    by delegating to the @ref SourceDetectionTask_ "detection" subtask.

    DetectCoaddSourcesTask is meant to be run after assembling a coadded image
    in a given band. The purpose of the task is to update the background,
    detect all sources in a single band and generate a set of parent
    footprints. Subsequent tasks in the multi-band processing procedure will
    merge sources across bands and, eventually, perform forced photometry.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`, optional
        Initial schema for the output catalog, modified-in place to include all
        fields set by this task.  If None, the source minimal schema will be used.
    **kwargs
        Additional keyword arguments.
    """

    _DefaultName = "detectCoaddForDiff"
    ConfigClass = DetectCoaddForDiffConfig

    def __init__(self, schema=None, **kwargs):
        # N.B. Super is used here to handle the multiple inheritance of PipelineTasks, the init tree
        # call structure has been reviewed carefully to be sure super will work as intended.

        super().__init__(**kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        # skyCoord needs errors in the schema
        lsst.afw.table.CoordKey.addErrorFields(self.schema)
        self.makeSubtask("detection", schema=self.schema)

        self.detectionSchema = afwTable.SourceCatalog(self.schema)
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask(
            "measurement", schema=self.schema, algMetadata=self.algMetadata
        )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        inputs["idFactory"] = idGenerator.make_table_id_factory()
        inputs["expId"] = idGenerator.catalog_id
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, exposure, idFactory, expId):
        """Run detection on an exposure.

        First scale the variance plane to match the observed variance
        using ``ScaleVarianceTask``. Then invoke the ``SourceDetectionTask_`` "detection" subtask to
        detect sources.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure on which to detect (may be backround-subtracted and scaled,
            depending on configuration).
        idFactory : `lsst.afw.table.IdFactory`
            IdFactory to set source identifiers.
        expId : `int`
            Exposure identifier (integer) for RNG seed.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``sources``
                Catalog of detections (`lsst.afw.table.SourceCatalog`).
            ``backgrounds``
                List of backgrounds (`list`).
        """
        table = afwTable.SourceTable.make(self.schema, idFactory)
        detections = self.detection.run(table, exposure, expId=expId)
        sources = detections.sources
        self.measurement.run(sources, exposure, exposureId=expId)
        return Struct(outputSources=sources, outputExposure=exposure)


class CoaddSubtractInputConnections(
    lsst.pipe.base.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    defaultTemplates={
        "templateCoaddName": "goodSeeing",
        "scienceCoaddName": "deepNightlyCoadd",
        "scienceCoaddSrcName": "deepNightlyCoadd_det",
    },
):
    template = connectionTypes.Input(
        doc="Input template from the coadd",
        dimensions=("tract", "patch", "skymap", "band"),
        storageClass="ExposureF",
        name="{templateCoaddName}Coadd",
    )
    science = connectionTypes.Input(
        doc="Input science exposure to subtract from.",
        dimensions=("tract", "patch", "skymap", "band", "day_obs"),
        storageClass="ExposureF",
        name="{scienceCoaddName}",
    )
    sources = connectionTypes.Input(
        doc="Sources measured on the science exposure; "
        "used to select sources for making the matching kernel.",
        dimensions=("tract", "patch", "skymap", "band", "day_obs"),
        storageClass="SourceCatalog",
        name="{scienceCoaddSrcName}",
    )


class CoaddSubtractImageOutputConnections(
    lsst.pipe.base.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    defaultTemplates={
        "templateCoaddName": "goodSeeing",
        "scienceCoaddName": "deepNightlyCoadd",
        "scienceCoaddSrcName": "deepNightlyCoadd_det",
    },
):
    difference = connectionTypes.Output(
        doc="Result of subtracting convolved template from science image.",
        dimensions=("tract", "patch", "skymap", "band", "day_obs"),
        storageClass="ExposureF",
        name="{templateCoaddName}Diff_nightlyDifferenceCoadd",
    )


class CoaddAlardLuptonSubtractConnections(
    CoaddSubtractInputConnections, CoaddSubtractImageOutputConnections
):
    def __init__(self, *, config=None):
        super().__init__(config=config)


class CoaddAlardLuptonSubtractConfig(
    AlardLuptonSubtractConfig, pipelineConnections=CoaddAlardLuptonSubtractConnections
):
    def setDefaults(self):
        super().setDefaults()
        self.doDecorrelation = False


class CoaddAlardLuptonSubtractTask(AlardLuptonSubtractTask):
    """Compute the image difference of a science and template image using
    the Alard & Lupton (1998) algorithm.
    """

    ConfigClass = CoaddAlardLuptonSubtractConfig
    _DefaultName = "CoaddAlardLuptonSubtract"


class DetectAndMeasureCoaddConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    defaultTemplates={
        "coaddName": "deepNightlyCoadd",
        "templateName": "goodSeeingCoadd",
        "differenceName": "goodSeeingDiff_nightlyDifferenceCoadd",
        "scienceSrcName": "deepNightlyCoadd_det",
        "warpTypeSuffix": "",
        "fakesType": "",
    },
):
    science = pipeBase.connectionTypes.Input(
        doc="Input science exposure.",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}_calexp",
    )
    matchedTemplate = pipeBase.connectionTypes.Input(
        doc="Warped and PSF-matched template used to create the difference image.",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="ExposureF",
        name="{fakesType}{templateName}",
    )
    difference = pipeBase.connectionTypes.Input(
        doc="Result of subtracting template from science.",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
        storageClass="ExposureF",
        name="{fakesType}{differenceName}",
    )
    outputSchema = pipeBase.connectionTypes.InitOutput(
        doc="Schema (as an example catalog) for output DIASource catalog.",
        storageClass="SourceCatalog",
        name="{fakesType}{differenceName}_diaSrc_schema",
    )
    diaSources = pipeBase.connectionTypes.Output(
        doc="Detected diaSources on the difference image.",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
        storageClass="SourceCatalog",
        name="{fakesType}{differenceName}_diaSrc",
    )
    subtractedMeasuredExposure = pipeBase.connectionTypes.Output(
        doc="Difference image with detection mask plane filled in.",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
        storageClass="ExposureF",
        name="{fakesType}{differenceName}Exp",
    )
    maskedStreaks = pipeBase.connectionTypes.Output(
        doc="Catalog of streak fit parameters for the difference image.",
        storageClass="ArrowNumpyDict",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
        name="{fakesType}{coaddName}Diff_streaks",
    )


class DetectAndMeasureCoaddConfig(
    DetectAndMeasureConfig, pipelineConnections=DetectAndMeasureCoaddConnections
):
    """Config for DetectAndMeasureTask"""

    idGenerator = SkyMapNightIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        self.measurement.plugins.names -= [
            "base_PeakLikelihoodFlux",
            "base_PeakCentroid",
        ]
        self.doApCorr = False
        self.idGenerator.packer.n_bands = None


class DetectAndMeasureCoaddTask(DetectAndMeasureTask):
    """Detect and measure sources on a difference image."""

    ConfigClass = DetectAndMeasureCoaddConfig
    _DefaultName = "detectAndMeasureCoadd"

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):
        inputs = butlerQC.get(inputRefs)
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        idFactory = idGenerator.make_table_id_factory()
        outputs = self.run(**inputs, idFactory=idFactory)
        butlerQC.put(outputs, outputRefs)


class TransformCoaddDiaSourceCatalogConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    defaultTemplates={
        "coaddName": "dsf",
        "differenceName": "goodSeeingDiff_nightlyDifferenceCoadd",
        "fakesType": "",
    },
):
    diaSourceSchema = connTypes.InitInput(
        doc="Schema for DIASource catalog output by ImageDifference.",
        storageClass="SourceCatalog",
        name="{fakesType}{differenceName}_diaSrc_schema",
    )
    diaSourceCat = connTypes.Input(
        doc="Catalog of DiaSources produced during image differencing.",
        name="{fakesType}{differenceName}_diaSrc",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    )
    diffIm = connTypes.Input(
        doc="Difference image on which the DiaSources were detected.",
        name="{fakesType}{differenceName}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    )
    # Don't know about this one, what shouhd the dimensions be?
    reliability = connTypes.Input(
        doc="Reliability (e.g. real/bogus) classificiation of diaSourceCat sources (optional).",
        name="{fakesType}{coaddName}RealBogusSources",
        storageClass="Catalog",
        dimensions=("instrument", "visit", "detector"),
    )
    diaSourceTable = connTypes.Output(
        doc=".",
        name="{fakesType}{differenceName}_diaSrcTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not self.config.doIncludeReliability:
            self.inputs.remove("reliability")


class TransformCoaddDiaSourceCatalogConfig(
    TransformDiaSourceCatalogConfig,
    pipelineConnections=TransformCoaddDiaSourceCatalogConnections,
):
    idGenerator = SkyMapNightIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        self.idGenerator.packer.n_bands = None


class TransformCoaddDiaSourceCatalogTask(TransformDiaSourceCatalogTask):
    """Transform a coadd DiaSource catalog by calibrating and renaming columns to
    produce a table ready to insert into the Apdb.
    Parameters
    ----------
    initInputs : `dict`
        Must contain ``diaSourceSchema`` as the schema for the input catalog.
    """

    ConfigClass = TransformCoaddDiaSourceCatalogConfig
    _DefaultName = "transformCoaddDiaSourceCatalog"
    # Needed to create a valid TransformCatalogBaseTask, but unused
    inputDataset = "goodSeeingDiff_nightlyDifferenceCoadd_diaSrc"
    outputDataset = "goodSeeingDiff_nightleDifferenceCoadd_diaSrcTable"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        inputs["skyId"] = idGenerator.catalog_id
        inputs["band"] = butlerQC.quantum.dataId["band"]
        inputs["idGenerator"] = idGenerator
        outputs = self.run(**inputs)

        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(self, diaSourceCat, diffIm, band, skyId, idGenerator, reliability=None):
        """Convert input catalog to ParquetTable/Pandas and run functors.

        Additionally, add new columns for stripping information from the
        exposure and into the DiaSource catalog.

        Parameters
        ----------
        diaSourceCat : `lsst.afw.table.SourceCatalog`
            Catalog of sources measured on the difference image.
        diffIm : `lsst.afw.image.Exposure`
            Result of subtracting template and science images.
        band : `str`
            Filter band of the science image.
        skyId : `int`
            Identifier for this patch+tract
        funcs : `lsst.pipe.tasks.functors.Functors`
            Functors to apply to the catalog's columns.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct with components.

            - ``diaSourceTable`` : Catalog of DiaSources with calibrated values
              and renamed columns.
              (`lsst.pipe.tasks.ParquetTable` or `pandas.DataFrame`)
        """

        self.log.info("Transforming/standardizing the DiaSource table skyId: %i", skyId)

        diaSourceDf = diaSourceCat.asAstropy().to_pandas()
        if self.config.doRemoveSkySources:
            diaSourceDf = diaSourceDf[~diaSourceDf["sky_source"]]
            diaSourceCat = diaSourceCat[~diaSourceCat["sky_source"]]

        data_id = idGenerator._bits.packer.unpack(skyId)

        diaSourceDf["time_processed"] = DateTime.now().toPython()
        diaSourceDf["snr"] = getSignificance(diaSourceCat)
        diaSourceDf["bboxSize"] = self.computeBBoxSizes(diaSourceCat)
        diaSourceDf["skyId"] = skyId
        diaSourceDf["day_obs"] = data_id["day_obs"]
        day_obs_str = str(data_id["day_obs"])
        time_str = day_obs_str[:4] + "-" + day_obs_str[4:6] + "-" + day_obs_str[6:8]
        date = Time(time_str)
        diaSourceDf["band"] = band
        diaSourceDf["midPointTai"] = date.tai
        diaSourceDf["diaObjectId"] = 0
        diaSourceDf["ssObjectId"] = 0

        if self.config.doIncludeReliability:
            reliabilityDf = reliability.asAstropy().to_pandas()
            # This uses the pandas index to match scores with diaSources
            # but it will silently fill with NaNs if they don't match.
            diaSourceDf = pd.merge(
                diaSourceDf, reliabilityDf, how="left", on="id", validate="1:1"
            )
            diaSourceDf = diaSourceDf.rename(columns={"score": "reliability"})
            if np.sum(diaSourceDf["reliability"].isna()) == len(diaSourceDf):
                self.log.warning("Reliability identifiers did not match diaSourceIds")
        else:
            diaSourceDf["reliability"] = np.float32(np.nan)

        if self.config.doPackFlags:
            # either bitpack the flags
            self.bitPackFlags(diaSourceDf)
        else:
            # or add the individual flag functors
            self.addUnpackedFlagFunctors()
            # and remove the packed flag functor
            if "flags" in self.funcs.funcDict:
                del self.funcs.funcDict["flags"]

        df = self.transform(band, diaSourceDf, self.funcs, dataId=None).df

        return pipeBase.Struct(
            diaSourceTable=df,
        )


class DrpCoaddAssociationPipeConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap"),
    defaultTemplates={
        "differenceName": "goodSeeingDiff_nightlyDifferenceCoadd",
        "fakesType": "",
    },
):
    diaSourceTables = pipeBase.connectionTypes.Input(
        doc="Set of catalogs of calibrated DiaSources.",
        name="{fakesType}{differenceName}_diaSrcTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch", "skymap", "band", "day_obs"),
        deferLoad=True,
        multiple=True,
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded "
        "exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    assocDiaSourceTable = pipeBase.connectionTypes.Output(
        doc="Catalog of DiaSources covering the patch and associated with a "
        "DiaObject.",
        name="{fakesType}{differenceName}_assocDiaSrcTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )
    diaObjectTable = pipeBase.connectionTypes.Output(
        doc="Catalog of DiaObjects created from spatially associating " "DiaSources.",
        name="{fakesType}{differenceName}_diaObjTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )


class DrpCoaddAssociationPipeConfig(
    DrpAssociationPipeConfig, pipelineConnections=DrpCoaddAssociationPipeConnections
):
    associator = pexConfig.ConfigurableField(
        target=SimpleAssociationCoaddTask,
        doc="Task used to associate DiaSources with DiaObjects.",
    )
    pass


class DrpCoaddAssociationPipeTask(DrpAssociationPipeTask):
    """Driver pipeline for loading DiaSource catalogs in a patch/tract
    region and associating them.
    """

    ConfigClass = DrpCoaddAssociationPipeConfig
    _DefaultName = "drpCoaddAssociation"

    def run(self, diaSourceTables, skyMap, tractId, patchId, idGenerator=None):
        """Trim DiaSources to the current Patch and run association.

        Takes in the set of DiaSource catalogs that covers the current patch,
        trims them to the dimensions of the patch, and [TODO: eventually]
        runs association on the concatenated DiaSource Catalog.

        Parameters
        ----------
        diaSourceTables : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            Set of DiaSource catalogs potentially covering this patch/tract.
        skyMap : `lsst.skymap.BaseSkyMap`
            SkyMap defining the patch/tract
        tractId : `int`
            Id of current tract being processed.
        patchId : `int`
            Id of current patch being processed.
        idGenerator : `lsst.meas.base.IdGenerator`, optional
            Object that generates Object IDs and random number generator seeds.

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``assocDiaSourceTable``
                Table of DiaSources with updated value for diaObjectId.
                (`pandas.DataFrame`)
            ``diaObjectTable``
                Table of DiaObjects from matching DiaSources
                (`pandas.DataFrame`).
        """
        self.log.info(
            "Running DPR Association on patch %i, tract %i...", patchId, tractId
        )

        skyInfo = makeSkyInfo(skyMap, tractId, patchId)

        # Get the patch bounding box.
        innerPatchBox = geom.Box2D(skyInfo.patchInfo.getInnerBBox())

        diaSourceHistory = []
        for catRef in diaSourceTables:
            cat = catRef.get()

            isInTractPatch = self._trimToPatch(cat, innerPatchBox, skyInfo.wcs)

            nDiaSrc = isInTractPatch.sum()
            self.log.info(
                "Read DiaSource catalog of length %i from %s."
                "Found %i sources within the patch/tract "
                "footprint.",
                len(cat),
                catRef.dataId,
                nDiaSrc,
            )

            if nDiaSrc <= 0:
                continue

            cutCat = cat[isInTractPatch]
            diaSourceHistory.append(cutCat)

        if diaSourceHistory:
            diaSourceHistoryCat = pd.concat(diaSourceHistory)
        else:
            # No rows to associate
            if self.config.doWriteEmptyTables:
                self.log.info("Constructing empty table")
                # Construct empty table using last table and dropping all the rows
                diaSourceHistoryCat = cat.drop(cat.index)
            else:
                raise pipeBase.NoWorkFound(
                    "Found no overlapping DIASources to associate."
                )

        self.log.info(
            "Found %i DiaSources overlapping patch %i, tract %i",
            len(diaSourceHistoryCat),
            patchId,
            tractId,
        )

        assocResult = self.associator.run(diaSourceHistoryCat, idGenerator=idGenerator)

        self.log.info(
            "Associated DiaSources into %i DiaObjects", len(assocResult.diaObjects)
        )

        if self.config.doAddDiaObjectCoords:
            assocResult.assocDiaSources = self._addDiaObjectCoords(
                assocResult.diaObjects, assocResult.assocDiaSources
            )

        return pipeBase.Struct(
            diaObjectTable=assocResult.diaObjects,
            assocDiaSourceTable=assocResult.assocDiaSources,
        )


class ForcedPhotCoaddFromDataFrameConnections(
    PipelineTaskConnections,
    dimensions=("band", "patch", "skymap", "tract", "day_obs"),
    defaultTemplates={
        "inputCoaddName": "goodSeeingDiff_nightlyDifferenceCoadd",
        "inputName": "deepNightlyCoadd_calexp",
    },
):
    refCat = cT.Input(
        doc="Catalog of positions at which to force photometry.",
        name="{inputCoaddName}_fullDiaObjTable",
        storageClass="DataFrame",
        dimensions=["skymap", "tract", "patch"],
        multiple=True,
        deferLoad=True,
    )
    exposure = cT.Input(
        doc="Input exposure to perform photometry on.",
        name="{inputName}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    )
    skyMap = cT.Input(
        doc="SkyMap dataset that defines the coordinate system of the reference catalog.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=["skymap"],
    )
    measCat = cT.Output(
        doc="Output forced photometry catalog.",
        name="forced_coadd_nightly_diaObject",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap", "day_obs"),
    )
    outputSchema = cT.InitOutput(
        doc="Schema for the output forced measurement catalogs.",
        name="forced_coadd_nightly_diaObject_schema",
        storageClass="SourceCatalog",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class ForcedPhotCoaddFromDataFrameConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ForcedPhotCoaddFromDataFrameConnections,
):
    measurement = lsst.pex.config.ConfigurableField(
        target=ForcedMeasurementTask, doc="subtask to do forced measurement"
    )
    coaddName = lsst.pex.config.Field(
        doc="coadd name: typically one of deep or goodSeeing",
        dtype=str,
        default="deep",
    )
    applyApCorr = lsst.pex.config.ConfigurableField(
        target=ApplyApCorrTask, doc="Subtask to apply aperture corrections"
    )
    catalogCalculation = lsst.pex.config.ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog",
    )
    psfFootprintScaling = lsst.pex.config.Field(
        dtype=float,
        doc="Scaling factor to apply to the PSF shape when footprintSource='psf' (ignored otherwise).",
        default=3.0,
    )
    idGenerator = SkyMapNightIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        self.measurement.doReplaceWithNoise = False
        # Only run a minimal set of plugins, as these measurements are only
        # needed for PSF-like sources.
        self.measurement.plugins.names = [
            "base_PixelFlags",
            "base_TransformedCentroidFromCoord",
            "base_PsfFlux",
            "base_LocalBackground",
            "base_LocalPhotoCalib",
            "base_LocalWcs",
        ]
        self.measurement.slots.shape = None
        # Keep track of which footprints contain streaks
        # self.measurement.plugins['base_PixelFlags'].masksFpAnywhere = ['STREAK']
        # self.measurement.plugins['base_PixelFlags'].masksFpCenter = ['STREAK']
        # Make catalogCalculation a no-op by default as no modelFlux is setup
        # by default in ForcedMeasurementTask.
        self.catalogCalculation.plugins.names = []

        self.measurement.copyColumns = {
            "id": "diaObjectId",
            "coord_ra": "coord_ra",
            "coord_dec": "coord_dec",
        }
        self.measurement.slots.centroid = "base_TransformedCentroidFromCoord"
        self.measurement.slots.psfFlux = "base_PsfFlux"
        self.idGenerator.packer.n_bands = None


class ForcedPhotCoaddFromDataFrameTask(pipeBase.PipelineTask):
    """Force Photometry on a per-detector exposure with coords from a DataFrame

    Uses input from a DataFrame instead of SourceCatalog
    like the base class ForcedPhotCcd does.
    Writes out a SourceCatalog so that the downstream
    WriteForcedSourceTableTask can be reused with output from this Task.
    """

    _DefaultName = "forcedPhotCoaddFromDataFrame"
    ConfigClass = ForcedPhotCoaddFromDataFrameConfig

    def __init__(self, refSchema=None, initInputs=None, **kwargs):
        # Parent's init assumes that we have a reference schema; Cannot reuse
        pipeBase.PipelineTask.__init__(self, **kwargs)

        self.makeSubtask(
            "measurement", refSchema=lsst.afw.table.SourceTable.makeMinimalSchema()
        )
        self.makeSubtask("catalogCalculation", schema=self.measurement.schema)
        self.outputSchema = lsst.afw.table.SourceCatalog(self.measurement.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        tract = butlerQC.quantum.dataId["tract"]
        skyMap = inputs.pop("skyMap")
        inputs["refWcs"] = skyMap[tract].getWcs()

        self.log.info(
            "Filtering ref cats: %s",
            ",".join([str(i.dataId) for i in inputs["refCat"]]),
        )
        if inputs["exposure"].getWcs() is not None:
            refCat = self.df2RefCat(
                [
                    i.get(parameters={"columns": ["diaObjectId", "ra", "dec"]})
                    for i in inputs["refCat"]
                ],
                inputs["exposure"].getBBox(),
                inputs["exposure"].getWcs(),
                inputs["exposure"],
            )
            inputs["refCat"] = refCat
            # generateMeasCat does not use the refWcs.
            inputs["measCat"], inputs["exposureId"] = self.generateMeasCat(
                inputRefs.exposure.dataId,
                inputs["exposure"],
                inputs["refCat"],
                inputs["refWcs"],
            )
            # attachFootprints only uses refWcs in ``transformed`` mode, which is not
            # supported in the DataFrame-backed task.
            valid_psf = self.attachPsfShapeFootprints(
                inputs["measCat"],
                inputs["exposure"],
                scaling=self.config.psfFootprintScaling,
            )
            # remove objects with no data
            inputs["measCat"] = inputs["measCat"].copy(deep=True)[
                valid_psf.astype(bool)
            ]

            outputs = self.run(**inputs)

            butlerQC.put(outputs, outputRefs)
        else:
            self.log.info(
                "No WCS for %s.  Skipping and no %s catalog will be written.",
                butlerQC.quantum.dataId,
                outputRefs.measCat.datasetType.name,
            )

    def generateMeasCat(self, dataId, exposure, refCat, refWcs):
        """Generate a measurement catalog.

        Parameters
        ----------
        dataId : `lsst.daf.butler.DataCoordinate`
            Butler data ID for this image, with ``{visit, detector}`` keys.
        exposure : `lsst.afw.image.exposure.Exposure`
            Exposure to generate the catalog for.
        refCat : `lsst.afw.table.SourceCatalog`
            Catalog of shapes and positions at which to force photometry.
        refWcs : `lsst.afw.image.SkyWcs`
            Reference world coordinate system.
            This parameter is not currently used.

        Returns
        -------
        measCat : `lsst.afw.table.SourceCatalog`
            Catalog of forced sources to measure.
        expId : `int`
            Unique binary id associated with the input exposure
        """
        id_generator = self.config.idGenerator.apply(dataId)
        measCat = self.measurement.generateMeasCat(
            exposure, refCat, refWcs, idFactory=id_generator.make_table_id_factory()
        )
        return measCat, id_generator.catalog_id

    def df2RefCat(self, dfList, exposureBBox, exposureWcs, exposure):
        """Convert list of DataFrames to reference catalog

        Concatenate list of DataFrames presumably from multiple patches and
        downselect rows that overlap the exposureBBox using the exposureWcs.

        Parameters
        ----------
        dfList : `list` of `pandas.DataFrame`
            Each element containst diaObjects with ra/dec position in degrees
            Columns 'diaObjectId', 'ra', 'dec' are expected
        exposureBBox :   `lsst.geom.Box2I`
            Bounding box on which to select rows that overlap
        exposureWcs : `lsst.afw.geom.SkyWcs`
            World coordinate system to convert sky coords in ref cat to
            pixel coords with which to compare with exposureBBox

        Returns
        -------
        refCat : `lsst.afw.table.SourceTable`
            Source Catalog with minimal schema that overlaps exposureBBox
        """
        df = pd.concat(dfList)
        # translate ra/dec coords in dataframe to detector pixel coords
        # to down select rows that overlap the detector bbox
        mapping = exposureWcs.getTransform().getMapping()
        x, y = mapping.applyInverse(
            np.array(df[["ra", "dec"]].values * 2 * np.pi / 360).T
        )
        inBBox = lsst.geom.Box2D(exposureBBox).contains(x, y)
        refCat = self.df2SourceCat(df[inBBox])
        return refCat

    def df2SourceCat(self, df):
        """Create minimal schema SourceCatalog from a pandas DataFrame.

        The forced measurement subtask expects this as input.

        Parameters
        ----------
        df : `pandas.DataFrame`
            DiaObjects with locations and ids.

        Returns
        -------
        outputCatalog : `lsst.afw.table.SourceTable`
            Output catalog with minimal schema.
        """
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        outputCatalog = lsst.afw.table.SourceCatalog(schema)
        outputCatalog.reserve(len(df))

        for diaObjectId, ra, dec in df[["ra", "dec"]].itertuples():
            outputRecord = outputCatalog.addNew()
            outputRecord.setId(diaObjectId)
            outputRecord.setCoord(lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees))
        return outputCatalog

    def attachPsfShapeFootprints(self, sources, exposure, scaling=3):
        """Attach Footprints to blank sources prior to measurement, by
        creating elliptical Footprints from the PSF moments.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog`
            Blank catalog (with all rows and columns, but values other than
            ``coord_ra``, ``coord_dec`` unpopulated).
            to which footprints should be attached.
        exposure : `lsst.afw.image.Exposure`
            Image object from which peak values and the PSF are obtained.
        scaling : `int`, optional
            Scaling factor to apply to the PSF second-moments ellipse in order
            to determine the footprint boundary.

        Notes
        -----
        This is a utility function for use by parent tasks; see
        `attachTransformedFootprints` for more information.
        """
        psf = exposure.getPsf()
        if psf is None:
            raise RuntimeError(
                "Cannot construct Footprints from PSF shape without a PSF."
            )
        bbox = exposure.getBBox()
        wcs = exposure.getWcs()
        valid_psf = np.zeros(len(sources), dtype=int)
        for i, record in enumerate(sources):
            localPoint = wcs.skyToPixel(record.getCoord())
            localIntPoint = lsst.geom.Point2I(localPoint)
            assert bbox.contains(localIntPoint), (
                f"Center for record {record.getId()} is not in exposure; this should be guaranteed by "
                "generateMeasCat."
            )

            try:
                ellipse = lsst.afw.geom.ellipses.Ellipse(
                    psf.computeShape(localPoint), localPoint
                )
            except:
                continue
            valid_psf[i] = 1
            ellipse.getCore().scale(scaling)
            spans = lsst.afw.geom.SpanSet.fromShape(ellipse)
            footprint = lsst.afw.detection.Footprint(spans.clippedTo(bbox), bbox)
            footprint.addPeak(
                localIntPoint.getX(),
                localIntPoint.getY(),
                exposure.image._get(localIntPoint, lsst.afw.image.PARENT),
            )
            record.setFootprint(footprint)
        return valid_psf

    def run(self, measCat, exposure, refCat, refWcs, exposureId=None):
        """Perform forced measurement on a single exposure.

        Parameters
        ----------
        measCat : `lsst.afw.table.SourceCatalog`
            The measurement catalog, based on the sources listed in the
            reference catalog.
        exposure : `lsst.afw.image.Exposure`
            The measurement image upon which to perform forced detection.
        refCat : `lsst.afw.table.SourceCatalog`
            The reference catalog of sources to measure.
        refWcs : `lsst.afw.image.SkyWcs`
            The WCS for the references.
        exposureId : `int`
            Optional unique exposureId used for random seed in measurement
            task.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Structure with fields:

            ``measCat``
                Catalog of forced measurement results
                (`lsst.afw.table.SourceCatalog`).
        """
        self.measurement.run(measCat, exposure, refCat, refWcs, exposureId=exposureId)
        self.catalogCalculation.run(measCat)

        return pipeBase.Struct(measCat=measCat)
