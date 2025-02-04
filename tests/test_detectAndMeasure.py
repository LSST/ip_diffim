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
import unittest

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom
from lsst.ip.diffim import detectAndMeasure, subtractImages
import lsst.meas.algorithms as measAlg
from lsst.pipe.base import InvalidQuantumError, UpstreamFailureNoWorkFound
import lsst.utils.tests

from utils import makeTestImage


class DetectAndMeasureTestBase:

    def _check_diaSource(self, refSources, diaSource, refIds=None,
                         matchDistance=1., scale=1., usePsfFlux=True,
                         rtol=0.025, atol=None):
        """Match a diaSource with a source in a reference catalog
        and compare properties.

        Parameters
        ----------
        refSources : `lsst.afw.table.SourceCatalog`
            The reference catalog.
        diaSource : `lsst.afw.table.SourceRecord`
            The new diaSource to match to the reference catalog.
        refIds : `list` of `int`, optional
            Source IDs of previously associated diaSources.
        matchDistance : `float`, optional
            Maximum distance allowed between the detected and reference source
            locations, in pixels.
        scale : `float`, optional
            Optional factor to scale the flux by before performing the test.
        usePsfFlux : `bool`, optional
            If set, test the PsfInstFlux field, otherwise use ApInstFlux.
        rtol : `float`, optional
            Relative tolerance of the flux value test.
        atol : `float`, optional
            Absolute tolerance of the flux value test.
        """
        distance = np.sqrt((diaSource.getX() - refSources.getX())**2
                           + (diaSource.getY() - refSources.getY())**2)
        self.assertLess(min(distance), matchDistance)
        src = refSources[np.argmin(distance)]
        if refIds is not None:
            # Check that the same source was not previously associated
            self.assertNotIn(src.getId(), refIds)
            refIds.append(src.getId())
        if atol is None:
            atol = rtol*src.getPsfInstFlux() if usePsfFlux else rtol*src.getApInstFlux()
        if usePsfFlux:
            self.assertFloatsAlmostEqual(src.getPsfInstFlux()*scale, diaSource.getPsfInstFlux(),
                                         rtol=rtol, atol=atol)
        else:
            self.assertFloatsAlmostEqual(src.getApInstFlux()*scale, diaSource.getApInstFlux(),
                                         rtol=rtol, atol=atol)

    def _check_values(self, values, minValue=None, maxValue=None):
        """Verify that an array has finite values, and optionally that they are
        within specified minimum and maximum bounds.

        Parameters
        ----------
        values : `numpy.ndarray`
            Array of values to check.
        minValue : `float`, optional
            Minimum allowable value.
        maxValue : `float`, optional
            Maximum allowable value.
        """
        self.assertTrue(np.all(np.isfinite(values)))
        if minValue is not None:
            self.assertTrue(np.all(values >= minValue))
        if maxValue is not None:
            self.assertTrue(np.all(values <= maxValue))

    def _setup_detection(self, doSkySources=False, nSkySources=5, **kwargs):
        """Setup and configure the detection and measurement PipelineTask.

        Parameters
        ----------
        doSkySources : `bool`, optional
            Generate sky sources.
        nSkySources : `int`, optional
            The number of sky sources to add in isolated background regions.
        **kwargs
            Any additional config parameters to set.

        Returns
        -------
        `lsst.pipe.base.PipelineTask`
            The configured Task to use for detection and measurement.
        """
        config = self.detectionTask.ConfigClass()
        config.doSkySources = doSkySources
        if doSkySources:
            config.skySources.nSources = nSkySources
        config.update(**kwargs)

        # Make a realistic id generator so that output catalog ids are useful.
        dataId = lsst.daf.butler.DataCoordinate.standardize(
            instrument="I",
            visit=42,
            detector=12,
            universe=lsst.daf.butler.DimensionUniverse(),
        )
        config.idGenerator.packer.name = "observation"
        config.idGenerator.packer["observation"].n_observations = 10000
        config.idGenerator.packer["observation"].n_detectors = 99
        config.idGenerator.n_releases = 8
        config.idGenerator.release_id = 2
        self.idGenerator = config.idGenerator.apply(dataId)

        return self.detectionTask(config=config)


class DetectAndMeasureTest(DetectAndMeasureTestBase, lsst.utils.tests.TestCase):
    detectionTask = detectAndMeasure.DetectAndMeasureTask

    def test_detection_xy0(self):
        """Basic functionality test with non-zero x0 and y0.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        fluxLevel = 500
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel, "x0": 12345, "y0": 67890}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)
        difference = science.clone()

        # Configure the detection Task
        detectionTask = self._setup_detection()

        # Run detection and check the results
        output = detectionTask.run(science, matchedTemplate, difference,
                                   idFactory=self.idGenerator.make_table_id_factory())
        subtractedMeasuredExposure = output.subtractedMeasuredExposure

        # Catalog ids should be very large from this id generator.
        self.assertTrue(all(output.diaSources['id'] > 1000000000))
        self.assertImagesEqual(subtractedMeasuredExposure.image, difference.image)

    def test_raise_bad_psf(self):
        """Detection should raise if the PSF width is NaN
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        fluxLevel = 500
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel, "x0": 12345, "y0": 67890}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)
        difference = science.clone()

        psfShape = difference.getPsf().computeBBox(lsst.geom.Point2D(0, 0)).getDimensions()
        psfcI = afwImage.ImageD(lsst.geom.Extent2I(psfShape[1], psfShape[0]))
        psfNew = np.zeros(psfShape)
        psfNew[0, :] = 1
        psfcI.array = psfNew
        psfcK = afwMath.FixedKernel(psfcI)
        psf = measAlg.KernelPsf(psfcK)
        difference.setPsf(psf)

        # Configure the detection Task
        detectionTask = self._setup_detection()

        # Run detection and check the results
        with self.assertRaises(UpstreamFailureNoWorkFound):
            detectionTask.run(science, matchedTemplate, difference)

    def test_measurements_finite(self):
        """Measured fluxes and centroids should always be finite.
        """
        columnNames = ["coord_ra", "coord_dec", "ip_diffim_forced_PsfFlux_instFlux"]

        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        transientSeed = 6
        xSize = 256
        ySize = 256
        kwargs = {"psfSize": 2.4, "x0": 0, "y0": 0,
                  "xSize": xSize, "ySize": ySize}
        science, sources = makeTestImage(seed=staticSeed, noiseLevel=noiseLevel, noiseSeed=6,
                                         nSrc=1, **kwargs)
        matchedTemplate, _ = makeTestImage(seed=staticSeed, noiseLevel=noiseLevel/4, noiseSeed=7,
                                           nSrc=1, **kwargs)
        rng = np.random.RandomState(3)
        xLoc = np.arange(-5, xSize+5, 10)
        rng.shuffle(xLoc)
        yLoc = np.arange(-5, ySize+5, 10)
        rng.shuffle(yLoc)
        transients, transientSources = makeTestImage(seed=transientSeed,
                                                     nSrc=len(xLoc), fluxLevel=1000.,
                                                     noiseLevel=noiseLevel, noiseSeed=8,
                                                     xLoc=xLoc, yLoc=yLoc,
                                                     **kwargs)
        difference = science.clone()
        difference.maskedImage -= matchedTemplate.maskedImage
        difference.maskedImage += transients.maskedImage

        # Configure the detection Task
        detectionTask = self._setup_detection(doForcedMeasurement=True)

        # Run detection and check the results
        output = detectionTask.run(science, matchedTemplate, difference)

        for column in columnNames:
            self._check_values(output.diaSources[column])
        self._check_values(output.diaSources.getX(), minValue=0, maxValue=xSize)
        self._check_values(output.diaSources.getY(), minValue=0, maxValue=ySize)
        self._check_values(output.diaSources.getPsfInstFlux())

    def test_raise_config_schema_mismatch(self):
        """Check that sources with specified flags are removed from the catalog.
        """
        # Configure the detection Task, and and set a config that is not in the schema
        with self.assertRaises(InvalidQuantumError):
            self._setup_detection(badSourceFlags=["Bogus_flag_42"])

    def test_remove_unphysical(self):
        """Check that sources with specified flags are removed from the catalog.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        xSize = 256
        ySize = 256
        kwargs = {"psfSize": 2.4, "xSize": xSize, "ySize": ySize}
        science, sources = makeTestImage(seed=staticSeed, noiseLevel=noiseLevel, noiseSeed=6,
                                         nSrc=1, **kwargs)
        matchedTemplate, _ = makeTestImage(seed=staticSeed, noiseLevel=noiseLevel/4, noiseSeed=7,
                                           nSrc=1, **kwargs)
        difference = science.clone()
        bbox = difference.getBBox()
        difference.maskedImage -= matchedTemplate.maskedImage

        # Configure the detection Task, and remove unphysical sources
        detectionTask = self._setup_detection(doForcedMeasurement=False, doSkySources=True, nSkySources=20,
                                              badSourceFlags=["base_PixelFlags_flag_offimage", ])

        # Run detection and check the results
        diaSources = detectionTask.run(science, matchedTemplate, difference).diaSources
        badDiaSrcDoRemove = ~bbox.contains(diaSources.getX(), diaSources.getY())
        nBadDoRemove = np.count_nonzero(badDiaSrcDoRemove)
        # Verify that all sources are physical
        self.assertEqual(nBadDoRemove, 0)
        # Set a few centroids outside the image bounding box
        nSetBad = 5
        for src in diaSources[0: nSetBad]:
            src["slot_Centroid_x"] += xSize
            src["slot_Centroid_y"] += ySize
            src["base_PixelFlags_flag_offimage"] = True
        # Verify that these sources are outside the image
        badDiaSrc = ~bbox.contains(diaSources.getX(), diaSources.getY())
        nBad = np.count_nonzero(badDiaSrc)
        self.assertEqual(nBad, nSetBad)
        diaSourcesNoBad = detectionTask._removeBadSources(diaSources)
        badDiaSrcNoBad = ~bbox.contains(diaSourcesNoBad.getX(), diaSourcesNoBad.getY())

        # Verify that no sources outside the image bounding box remain
        self.assertEqual(np.count_nonzero(badDiaSrcNoBad), 0)
        self.assertEqual(len(diaSourcesNoBad), len(diaSources) - nSetBad)

    def test_detect_transients(self):
        """Run detection on a difference image containing transients.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        transientSeed = 6
        fluxLevel = 500
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)

        # Configure the detection Task
        detectionTask = self._setup_detection(doMerge=False)
        kwargs["seed"] = transientSeed
        kwargs["nSrc"] = 10
        kwargs["fluxLevel"] = 1000

        blendedKwargs = kwargs
        blendedKwargs["nSrc"] = 1

        # Run detection and check the results
        def _detection_wrapper(positive=True):
            transients, transientSources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=8, **kwargs)
            # Make a nearby transient source that doesn't fall off the image
            # so we can test deblending
            transientsBbox = transients.getBBox()
            oneSourceX = transientSources[0].getX()
            oneSourceY = transientSources[0].getY()
            if oneSourceX < transientsBbox.getCenter()[0]:
                blendedSourceX = oneSourceX + 10
            else:
                blendedSourceX = oneSourceX - 10
            if oneSourceY < transientsBbox.getCenter()[1]:
                blendedSourceY = oneSourceY + 10
            else:
                blendedSourceY = oneSourceY - 10
            blendedTransients, blendedTransientSources = makeTestImage(noiseLevel=0,
                                                                       noiseSeed=8,
                                                                       xLoc=[blendedSourceX,],
                                                                       yLoc=[blendedSourceY,],
                                                                       **blendedKwargs)
            blendedTransientSources["id"][0] = np.max(transientSources["id"]) + 1
            transientSources.extend(blendedTransientSources)
            transients.maskedImage += blendedTransients.maskedImage
            difference = science.clone()
            difference.maskedImage -= matchedTemplate.maskedImage
            if positive:
                difference.maskedImage += transients.maskedImage
            else:
                difference.maskedImage -= transients.maskedImage
            # NOTE: NoiseReplacer (run by forcedMeasurement) can modify the
            # science image if we've e.g. removed parents post-deblending.
            # Pass a clone of the science image, so that it doesn't disrupt
            # later tests.
            output = detectionTask.run(science.clone(), matchedTemplate, difference)
            refIds = []
            scale = 1. if positive else -1.
            for diaSource in output.diaSources:
                self._check_diaSource(transientSources.copy(deep=True), diaSource, refIds=refIds, scale=scale)
        _detection_wrapper(positive=True)
        _detection_wrapper(positive=False)

    def test_missing_mask_planes(self):
        """Check that detection runs with missing mask planes.
        """
        # Set up the simulated images
        noiseLevel = 1.
        fluxLevel = 500
        kwargs = {"psfSize": 2.4, "fluxLevel": fluxLevel, "addMaskPlanes": []}
        # Use different seeds for the science and template so every source is a diaSource
        science, sources = makeTestImage(seed=5, noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(seed=6, noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)

        difference = science.clone()
        difference.maskedImage -= matchedTemplate.maskedImage
        detectionTask = self._setup_detection()

        # Verify that detection runs without errors
        detectionTask.run(science, matchedTemplate, difference)

    def test_detect_dipoles(self):
        """Run detection on a difference image containing dipoles.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        fluxLevel = 1000
        fluxRange = 1.5
        nSources = 10
        offset = 1
        xSize = 300
        ySize = 300
        kernelSize = 32
        # Avoid placing sources near the edge for this test, so that we can
        # easily check that the correct number of sources are detected.
        templateBorderSize = kernelSize//2
        dipoleFlag = "ip_diffim_DipoleFit_classification"
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel, "fluxRange": fluxRange,
                  "nSrc": nSources, "templateBorderSize": templateBorderSize, "kernelSize": kernelSize,
                  "xSize": xSize, "ySize": ySize}
        dipoleFlag = "ip_diffim_DipoleFit_classification"
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)
        difference = science.clone()
        matchedTemplate.image.array[...] = np.roll(matchedTemplate.image.array[...], offset, axis=0)
        matchedTemplate.variance.array[...] = np.roll(matchedTemplate.variance.array[...], offset, axis=0)
        matchedTemplate.mask.array[...] = np.roll(matchedTemplate.mask.array[...], offset, axis=0)
        difference.maskedImage -= matchedTemplate.maskedImage[science.getBBox()]

        detectionTask = self._setup_detection(doMerge=True)
        output = detectionTask.run(science, matchedTemplate, difference)
        self.assertEqual(len(output.diaSources), len(sources))
        refIds = []
        for diaSource in output.diaSources:
            if diaSource[dipoleFlag]:
                self._check_diaSource(sources, diaSource, refIds=refIds, scale=0,
                                      rtol=0.05, atol=None, usePsfFlux=False)
                self.assertFloatsAlmostEqual(diaSource["ip_diffim_DipoleFit_orientation"],
                                             -np.pi / 2, atol=2.)
                self.assertFloatsAlmostEqual(diaSource["ip_diffim_DipoleFit_separation"], offset, rtol=0.1)
            else:
                raise ValueError("DiaSource with ID %s is not a dipole!", diaSource.getId())

    def test_sky_sources(self):
        """Add sky sources and check that they are sufficiently far from other
        sources and have negligible flux.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        transientSeed = 6
        transientFluxLevel = 1000.
        transientFluxRange = 1.5
        fluxLevel = 500
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)
        transients, transientSources = makeTestImage(seed=transientSeed, psfSize=2.4,
                                                     nSrc=10, fluxLevel=transientFluxLevel,
                                                     fluxRange=transientFluxRange,
                                                     noiseLevel=noiseLevel, noiseSeed=8)
        difference = science.clone()
        difference.maskedImage -= matchedTemplate.maskedImage
        difference.maskedImage += transients.maskedImage
        kernelWidth = np.max(science.psf.getKernel().getDimensions())//2

        # Configure the detection Task
        detectionTask = self._setup_detection(doSkySources=True)

        # Run detection and check the results
        output = detectionTask.run(science, matchedTemplate, difference,
                                   idFactory=self.idGenerator.make_table_id_factory())
        skySources = output.diaSources[output.diaSources["sky_source"]]
        self.assertEqual(len(skySources), detectionTask.config.skySources.nSources)
        for skySource in skySources:
            # The sky sources should not be close to any other source
            with self.assertRaises(AssertionError):
                self._check_diaSource(transientSources, skySource, matchDistance=kernelWidth)
            with self.assertRaises(AssertionError):
                self._check_diaSource(sources, skySource, matchDistance=kernelWidth)
            # The sky sources should have low flux levels.
            self._check_diaSource(transientSources, skySource, matchDistance=1000, scale=0.,
                                  atol=np.sqrt(transientFluxRange*transientFluxLevel))

        # Catalog ids should be very large from this id generator.
        self.assertTrue(all(output.diaSources['id'] > 1000000000))

    def test_exclude_mask_detections(self):
        """Sources with certain bad mask planes set should not be detected.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        transientSeed = 6
        fluxLevel = 500
        radius = 2
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)

        _checkMask = subtractImages.AlardLuptonSubtractTask._checkMask
        # Configure the detection Task
        detectionTask = self._setup_detection()
        excludeMaskPlanes = detectionTask.config.detection.excludeMaskPlanes
        nBad = len(excludeMaskPlanes)
        self.assertGreater(nBad, 0)
        kwargs["seed"] = transientSeed
        kwargs["nSrc"] = nBad
        kwargs["fluxLevel"] = 1000

        # Run detection and check the results
        def _detection_wrapper(setFlags=True):
            transients, transientSources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=8, **kwargs)
            difference = science.clone()
            difference.maskedImage -= matchedTemplate.maskedImage
            difference.maskedImage += transients.maskedImage
            if setFlags:
                for src, badMask in zip(transientSources, excludeMaskPlanes):
                    srcX = int(src.getX())
                    srcY = int(src.getY())
                    srcBbox = lsst.geom.Box2I(lsst.geom.Point2I(srcX - radius, srcY - radius),
                                              lsst.geom.Extent2I(2*radius + 1, 2*radius + 1))
                    difference[srcBbox].mask.array |= lsst.afw.image.Mask.getPlaneBitMask(badMask)
            output = detectionTask.run(science, matchedTemplate, difference)
            refIds = []
            goodSrcFlags = _checkMask(difference.mask, transientSources, excludeMaskPlanes)
            if setFlags:
                self.assertEqual(np.sum(~goodSrcFlags), nBad)
            else:
                self.assertEqual(np.sum(~goodSrcFlags), 0)
            for diaSource, goodSrcFlag in zip(output.diaSources, goodSrcFlags):
                if ~goodSrcFlag:
                    with self.assertRaises(AssertionError):
                        self._check_diaSource(transientSources, diaSource, refIds=refIds)
                else:
                    self._check_diaSource(transientSources, diaSource, refIds=refIds)
        _detection_wrapper(setFlags=False)
        _detection_wrapper(setFlags=True)

    def test_fake_mask_plane_propagation(self):
        """Test that we have the mask planes related to fakes in diffim images.
        This is testing method called updateMasks
        """
        xSize = 256
        ySize = 256
        science, sources = makeTestImage(psfSize=2.4, xSize=xSize, ySize=ySize, doApplyCalibration=True)
        science_fake_img, science_fake_sources = makeTestImage(
            psfSize=2.4, xSize=xSize, ySize=ySize, seed=5, nSrc=3, noiseLevel=0.25, fluxRange=1
        )
        template, _ = makeTestImage(psfSize=2.4, xSize=xSize, ySize=ySize, doApplyCalibration=True)
        tmplt_fake_img, tmplt_fake_sources = makeTestImage(
            psfSize=2.4, xSize=xSize, ySize=ySize, seed=9, nSrc=3, noiseLevel=0.25, fluxRange=1
        )
        # created fakes and added them to the images
        science.image += science_fake_img.image
        template.image += tmplt_fake_img.image

        # TODO: DM-40796 update to INJECTED names when source injection gets refactored
        # adding mask planes to both science and template images
        science.mask.addMaskPlane("FAKE")
        science_fake_bitmask = science.mask.getPlaneBitMask("FAKE")
        template.mask.addMaskPlane("FAKE")
        template_fake_bitmask = template.mask.getPlaneBitMask("FAKE")

        # makeTestImage sets the DETECTED plane on the sources; we can use
        # that to set the FAKE plane on the science and template images.
        detected = science_fake_img.mask.getPlaneBitMask("DETECTED")
        fake_pixels = (science_fake_img.mask.array & detected).nonzero()
        science.mask.array[fake_pixels] |= science_fake_bitmask
        detected = tmplt_fake_img.mask.getPlaneBitMask("DETECTED")
        fake_pixels = (tmplt_fake_img.mask.array & detected).nonzero()
        template.mask.array[fake_pixels] |= science_fake_bitmask

        science_fake_masked = (science.mask.array & science_fake_bitmask) > 0
        template_fake_masked = (template.mask.array & template_fake_bitmask) > 0

        subtractConfig = subtractImages.AlardLuptonSubtractTask.ConfigClass()
        subtractConfig.sourceSelector.signalToNoise.fluxField = "truth_instFlux"
        subtractConfig.sourceSelector.signalToNoise.errField = "truth_instFluxErr"
        subtractTask = subtractImages.AlardLuptonSubtractTask(config=subtractConfig)
        subtraction = subtractTask.run(template, science, sources)

        # check subtraction mask plane is set where we set the previous masks
        diff_mask = subtraction.difference.mask

        # science mask should be now in INJECTED
        inj_masked = (diff_mask.array & diff_mask.getPlaneBitMask("INJECTED")) > 0

        # template mask should be now in INJECTED_TEMPLATE
        injTmplt_masked = (diff_mask.array & diff_mask.getPlaneBitMask("INJECTED_TEMPLATE")) > 0

        self.assertFloatsEqual(inj_masked.astype(int), science_fake_masked.astype(int))
        # The template is convolved, so the INJECTED_TEMPLATE mask plane may
        # include more pixels than the FAKE mask plane
        injTmplt_masked &= template_fake_masked
        self.assertFloatsEqual(injTmplt_masked.astype(int), template_fake_masked.astype(int))

        # Now check that detection of fakes have the correct flag for injections
        detectionTask = self._setup_detection()

        output = detectionTask.run(subtraction.matchedScience,
                                   subtraction.matchedTemplate,
                                   subtraction.difference)

        sci_refIds = []
        tmpl_refIds = []
        for diaSrc in output.diaSources:
            if diaSrc['base_PsfFlux_instFlux'] > 0:
                self._check_diaSource(science_fake_sources, diaSrc, scale=1, refIds=sci_refIds)
                self.assertTrue(diaSrc['base_PixelFlags_flag_injected'])
                self.assertTrue(diaSrc['base_PixelFlags_flag_injectedCenter'])
                self.assertFalse(diaSrc['base_PixelFlags_flag_injected_template'])
                self.assertFalse(diaSrc['base_PixelFlags_flag_injected_templateCenter'])
            else:
                self._check_diaSource(tmplt_fake_sources, diaSrc, scale=-1, refIds=tmpl_refIds)
                self.assertTrue(diaSrc['base_PixelFlags_flag_injected_template'])
                self.assertTrue(diaSrc['base_PixelFlags_flag_injected_templateCenter'])
                self.assertFalse(diaSrc['base_PixelFlags_flag_injected'])
                self.assertFalse(diaSrc['base_PixelFlags_flag_injectedCenter'])

    def test_mask_streaks(self):
        """Run detection on a difference image containing a streak.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        fluxLevel = 500
        xSize = 400
        ySize = 400
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel, "xSize": xSize, "ySize": ySize}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)

        # Configure the detection Task
        detectionTask = self._setup_detection(doMerge=False, doMaskStreaks=True)

        # Test that no streaks are detected
        difference = science.clone()
        difference.maskedImage -= matchedTemplate.maskedImage
        output = detectionTask.run(science, matchedTemplate, difference)
        outMask = output.subtractedMeasuredExposure.mask.array
        streakMask = output.subtractedMeasuredExposure.mask.getPlaneBitMask("STREAK")
        streakMaskSet = (outMask & streakMask) > 0
        self.assertTrue(np.all(streakMaskSet == 0))

        # Add streak-like shape and check that streak is detected
        difference.image.array[20:23, 40:200] += 50
        output = detectionTask.run(science, matchedTemplate, difference)
        outMask = output.subtractedMeasuredExposure.mask.array
        streakMask = output.subtractedMeasuredExposure.mask.getPlaneBitMask("STREAK")
        streakMaskSet = (outMask & streakMask) > 0
        self.assertTrue(np.all(streakMaskSet[20:23, 40:200]))

        # Add a more trapezoid shaped streak across an image that is
        # fainter and check that it is also detected
        bbox = science.getBBox()
        difference = science.clone()
        difference.maskedImage -= matchedTemplate.maskedImage
        width = 100
        amplitude = 5
        x0 = -100 + bbox.getBeginX()
        y0 = -100 + bbox.getBeginY()
        x1 = xSize + x0 + 100
        y1 = ySize/2
        corner_coords = [lsst.geom.Point2D(x0, y0),
                         lsst.geom.Point2D(x0, y0 + width),
                         lsst.geom.Point2D(x1, y1),
                         lsst.geom.Point2D(x1 + width, y1)]
        streak_trapezoid = afwGeom.Polygon(corner_coords)
        streak_image = streak_trapezoid.createImage(bbox)
        streak_image *= amplitude
        difference.image.array += streak_image.array
        output = detectionTask.run(science, matchedTemplate, difference)
        outMask = output.subtractedMeasuredExposure.mask.array
        streakMask = output.subtractedMeasuredExposure.mask.getPlaneBitMask("STREAK")
        streakMaskSet = (outMask & streakMask) > 0
        # Check that pixel values in streak_image equal the streak amplitude
        streak_check = np.where(streak_image.array == amplitude)
        self.assertTrue(np.all(streakMaskSet[streak_check]))
        # Check that the entire image was not masked STREAK
        self.assertFalse(np.all(streakMaskSet))


class DetectAndMeasureScoreTest(DetectAndMeasureTestBase, lsst.utils.tests.TestCase):
    detectionTask = detectAndMeasure.DetectAndMeasureScoreTask

    def test_detection_xy0(self):
        """Basic functionality test with non-zero x0 and y0.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        fluxLevel = 500
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel, "x0": 12345, "y0": 67890}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)
        difference = science.clone()
        subtractTask = subtractImages.AlardLuptonPreconvolveSubtractTask()
        scienceKernel = science.psf.getKernel()
        score = subtractTask._convolveExposure(difference, scienceKernel, subtractTask.convolutionControl)

        # Configure the detection Task
        detectionTask = self._setup_detection()

        # Run detection and check the results
        output = detectionTask.run(science, matchedTemplate, difference, score,
                                   idFactory=self.idGenerator.make_table_id_factory())

        # Catalog ids should be very large from this id generator.
        self.assertTrue(all(output.diaSources['id'] > 1000000000))
        subtractedMeasuredExposure = output.subtractedMeasuredExposure

        self.assertImagesEqual(subtractedMeasuredExposure.image, difference.image)

    def test_measurements_finite(self):
        """Measured fluxes and centroids should always be finite.
        """
        columnNames = ["coord_ra", "coord_dec", "ip_diffim_forced_PsfFlux_instFlux"]

        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        transientSeed = 6
        xSize = 256
        ySize = 256
        kwargs = {"psfSize": 2.4, "x0": 0, "y0": 0,
                  "xSize": xSize, "ySize": ySize}
        science, sources = makeTestImage(seed=staticSeed, noiseLevel=noiseLevel, noiseSeed=6,
                                         nSrc=1, **kwargs)
        matchedTemplate, _ = makeTestImage(seed=staticSeed, noiseLevel=noiseLevel/4, noiseSeed=7,
                                           nSrc=1, **kwargs)
        rng = np.random.RandomState(3)
        xLoc = np.arange(-5, xSize+5, 10)
        rng.shuffle(xLoc)
        yLoc = np.arange(-5, ySize+5, 10)
        rng.shuffle(yLoc)
        transients, transientSources = makeTestImage(seed=transientSeed,
                                                     nSrc=len(xLoc), fluxLevel=1000.,
                                                     noiseLevel=noiseLevel, noiseSeed=8,
                                                     xLoc=xLoc, yLoc=yLoc,
                                                     **kwargs)
        difference = science.clone()
        difference.maskedImage -= matchedTemplate.maskedImage
        difference.maskedImage += transients.maskedImage
        subtractTask = subtractImages.AlardLuptonPreconvolveSubtractTask()
        scienceKernel = science.psf.getKernel()
        score = subtractTask._convolveExposure(difference, scienceKernel, subtractTask.convolutionControl)

        # Configure the detection Task
        detectionTask = self._setup_detection(doForcedMeasurement=True)

        # Run detection and check the results
        output = detectionTask.run(science, matchedTemplate, difference, score)

        for column in columnNames:
            self._check_values(output.diaSources[column])
        self._check_values(output.diaSources.getX(), minValue=0, maxValue=xSize)
        self._check_values(output.diaSources.getY(), minValue=0, maxValue=ySize)
        self._check_values(output.diaSources.getPsfInstFlux())

    def test_detect_transients(self):
        """Run detection on a difference image containing transients.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        transientSeed = 6
        fluxLevel = 500
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)
        scienceKernel = science.psf.getKernel()
        subtractTask = subtractImages.AlardLuptonPreconvolveSubtractTask()

        # Configure the detection Task
        detectionTask = self._setup_detection(doMerge=False)
        kwargs["seed"] = transientSeed
        kwargs["nSrc"] = 10
        kwargs["fluxLevel"] = 1000

        # Run detection and check the results
        def _detection_wrapper(positive=True):
            """Simulate positive or negative transients and run detection.

            Parameters
            ----------
            positive : `bool`, optional
                If set, use positive transient sources.
            """

            transients, transientSources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=8, **kwargs)
            difference = science.clone()
            difference.maskedImage -= matchedTemplate.maskedImage
            if positive:
                difference.maskedImage += transients.maskedImage
            else:
                difference.maskedImage -= transients.maskedImage
            score = subtractTask._convolveExposure(difference, scienceKernel, subtractTask.convolutionControl)
            # NOTE: NoiseReplacer (run by forcedMeasurement) can modify the
            # science image if we've e.g. removed parents post-deblending.
            # Pass a clone of the science image, so that it doesn't disrupt
            # later tests.
            output = detectionTask.run(science.clone(), matchedTemplate, difference, score)
            refIds = []
            scale = 1. if positive else -1.
            # sources near the edge may have untrustworthy centroids
            goodSrcFlags = ~output.diaSources['base_PixelFlags_flag_edge']
            for diaSource, goodSrcFlag in zip(output.diaSources, goodSrcFlags):
                if goodSrcFlag:
                    self._check_diaSource(transientSources, diaSource, refIds=refIds, scale=scale)
        _detection_wrapper(positive=True)
        _detection_wrapper(positive=False)

    def test_detect_dipoles(self):
        """Run detection on a difference image containing dipoles.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        fluxLevel = 1000
        fluxRange = 1.5
        nSources = 10
        offset = 1
        xSize = 300
        ySize = 300
        kernelSize = 32
        # Avoid placing sources near the edge for this test, so that we can
        # easily check that the correct number of sources are detected.
        templateBorderSize = kernelSize//2
        dipoleFlag = "ip_diffim_DipoleFit_classification"
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel, "fluxRange": fluxRange,
                  "nSrc": nSources, "templateBorderSize": templateBorderSize, "kernelSize": kernelSize,
                  "xSize": xSize, "ySize": ySize}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)
        difference = science.clone()
        # Shift the template by a pixel in order to make dipoles in the difference image.
        matchedTemplate.image.array[...] = np.roll(matchedTemplate.image.array[...], offset, axis=0)
        matchedTemplate.variance.array[...] = np.roll(matchedTemplate.variance.array[...], offset, axis=0)
        matchedTemplate.mask.array[...] = np.roll(matchedTemplate.mask.array[...], offset, axis=0)
        difference.maskedImage -= matchedTemplate.maskedImage[science.getBBox()]
        subtractTask = subtractImages.AlardLuptonPreconvolveSubtractTask()
        scienceKernel = science.psf.getKernel()
        score = subtractTask._convolveExposure(difference, scienceKernel, subtractTask.convolutionControl)

        detectionTask = self._setup_detection()
        output = detectionTask.run(science, matchedTemplate, difference, score)
        self.assertEqual(len(output.diaSources), len(sources))
        refIds = []
        for diaSource in output.diaSources:
            if diaSource[dipoleFlag]:
                self._check_diaSource(sources, diaSource, refIds=refIds, scale=0,
                                      rtol=0.05, atol=None, usePsfFlux=False)
                self.assertFloatsAlmostEqual(diaSource["ip_diffim_DipoleFit_orientation"],
                                             -np.pi / 2, atol=2.)
                self.assertFloatsAlmostEqual(diaSource["ip_diffim_DipoleFit_separation"], offset, rtol=0.1)
            else:
                raise ValueError("DiaSource with ID %s is not a dipole!", diaSource.getId())

    def test_sky_sources(self):
        """Add sky sources and check that they are sufficiently far from other
        sources and have negligible flux.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        transientSeed = 6
        transientFluxLevel = 1000.
        transientFluxRange = 1.5
        fluxLevel = 500
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)
        transients, transientSources = makeTestImage(seed=transientSeed, psfSize=2.4,
                                                     nSrc=10, fluxLevel=transientFluxLevel,
                                                     fluxRange=transientFluxRange,
                                                     noiseLevel=noiseLevel, noiseSeed=8)
        difference = science.clone()
        difference.maskedImage -= matchedTemplate.maskedImage
        difference.maskedImage += transients.maskedImage
        subtractTask = subtractImages.AlardLuptonPreconvolveSubtractTask()
        scienceKernel = science.psf.getKernel()
        kernelWidth = np.max(scienceKernel.getDimensions())//2
        score = subtractTask._convolveExposure(difference, scienceKernel, subtractTask.convolutionControl)

        # Configure the detection Task
        detectionTask = self._setup_detection(doSkySources=True)

        # Run detection and check the results
        output = detectionTask.run(science, matchedTemplate, difference, score,
                                   idFactory=self.idGenerator.make_table_id_factory())
        nSkySourcesGenerated = detectionTask.metadata["n_skySources"]
        skySources = output.diaSources[output.diaSources["sky_source"]]
        self.assertEqual(len(skySources), nSkySourcesGenerated)
        for skySource in skySources:
            # The sky sources should not be close to any other source
            with self.assertRaises(AssertionError):
                self._check_diaSource(transientSources, skySource, matchDistance=kernelWidth)
            with self.assertRaises(AssertionError):
                self._check_diaSource(sources, skySource, matchDistance=kernelWidth)
            # The sky sources should have low flux levels.
            self._check_diaSource(transientSources, skySource, matchDistance=1000, scale=0.,
                                  atol=np.sqrt(transientFluxRange*transientFluxLevel))

        # Catalog ids should be very large from this id generator.
        self.assertTrue(all(output.diaSources['id'] > 1000000000))

    def test_exclude_mask_detections(self):
        """Sources with certain bad mask planes set should not be detected.
        """
        # Set up the simulated images
        noiseLevel = 1.
        staticSeed = 1
        transientSeed = 6
        fluxLevel = 500
        radius = 2
        kwargs = {"seed": staticSeed, "psfSize": 2.4, "fluxLevel": fluxLevel}
        science, sources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=6, **kwargs)
        matchedTemplate, _ = makeTestImage(noiseLevel=noiseLevel/4, noiseSeed=7, **kwargs)

        subtractTask = subtractImages.AlardLuptonPreconvolveSubtractTask()
        scienceKernel = science.psf.getKernel()
        # Configure the detection Task
        detectionTask = self._setup_detection()
        excludeMaskPlanes = detectionTask.config.detection.excludeMaskPlanes
        nBad = len(excludeMaskPlanes)
        self.assertGreater(nBad, 0)
        kwargs["seed"] = transientSeed
        kwargs["nSrc"] = nBad
        kwargs["fluxLevel"] = 1000

        # Run detection and check the results
        def _detection_wrapper(setFlags=True):
            transients, transientSources = makeTestImage(noiseLevel=noiseLevel, noiseSeed=8, **kwargs)
            difference = science.clone()
            difference.maskedImage -= matchedTemplate.maskedImage
            difference.maskedImage += transients.maskedImage
            if setFlags:
                for src, badMask in zip(transientSources, excludeMaskPlanes):
                    srcX = int(src.getX())
                    srcY = int(src.getY())
                    srcBbox = lsst.geom.Box2I(lsst.geom.Point2I(srcX - radius, srcY - radius),
                                              lsst.geom.Extent2I(2*radius + 1, 2*radius + 1))
                    difference[srcBbox].mask.array |= lsst.afw.image.Mask.getPlaneBitMask(badMask)
            score = subtractTask._convolveExposure(difference, scienceKernel, subtractTask.convolutionControl)
            output = detectionTask.run(science, matchedTemplate, difference, score)
            refIds = []
            goodSrcFlags = subtractTask._checkMask(difference.mask, transientSources, excludeMaskPlanes)
            if setFlags:
                self.assertEqual(np.sum(~goodSrcFlags), nBad)
            else:
                self.assertEqual(np.sum(~goodSrcFlags), 0)
            for diaSource, goodSrcFlag in zip(output.diaSources, goodSrcFlags):
                if ~goodSrcFlag:
                    with self.assertRaises(AssertionError):
                        self._check_diaSource(transientSources, diaSource, refIds=refIds)
                else:
                    self._check_diaSource(transientSources, diaSource, refIds=refIds)
        _detection_wrapper(setFlags=False)
        _detection_wrapper(setFlags=True)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
