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

import unittest

from astropy import units as u

import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.algorithms as measAlg
from lsst.ip.diffim import subtractImages
from lsst.pex.config import FieldValidationError
from lsst.pipe.base import NoWorkFound
import lsst.utils.tests
import numpy as np
from lsst.ip.diffim.utils import (computeRobustStatistics, computePSFNoiseEquivalentArea,
                                  evaluateMeanPsfFwhm, getPsfFwhm)
from lsst.pex.exceptions import InvalidParameterError

from utils import makeStats, makeTestImage, CustomCoaddPsf


class AlardLuptonSubtractTestBase:
    def _setup_subtraction(self, **kwargs):
        """Setup and configure the image subtraction PipelineTask.

        Parameters
        ----------
        **kwargs
            Any additional config parameters to set.

        Returns
        -------
        `lsst.pipe.base.PipelineTask`
            The configured Task to use for detection and measurement.
        """
        config = self.subtractTask.ConfigClass()
        config.doSubtractBackground = False
        config.sourceSelector.signalToNoise.fluxField = "truth_instFlux"
        config.sourceSelector.signalToNoise.errField = "truth_instFluxErr"
        config.sourceSelector.doUnresolved = True
        config.sourceSelector.doIsolated = True
        config.sourceSelector.doRequirePrimary = True
        config.sourceSelector.doFlags = True
        config.sourceSelector.doSignalToNoise = True
        config.sourceSelector.flags.bad = ["base_PsfFlux_flag", ]
        config.update(**kwargs)

        return self.subtractTask(config=config)


class AlardLuptonSubtractTest(AlardLuptonSubtractTestBase, lsst.utils.tests.TestCase):
    subtractTask = subtractImages.AlardLuptonSubtractTask

    def test_allowed_config_modes(self):
        """Verify the allowable modes for convolution.
        """
        config = subtractImages.AlardLuptonSubtractTask.ConfigClass()
        config.mode = 'auto'
        config.mode = 'convolveScience'
        config.mode = 'convolveTemplate'

        with self.assertRaises(FieldValidationError):
            config.mode = 'aotu'

    def test_mismatched_template(self):
        """Test that an error is raised if the template
        does not fully contain the science image.
        """
        xSize = 200
        ySize = 200
        science, sources = makeTestImage(psfSize=2.4, xSize=xSize + 20, ySize=ySize + 20)
        template, _ = makeTestImage(psfSize=2.4, xSize=xSize, ySize=ySize, doApplyCalibration=True)
        task = self._setup_subtraction()
        with self.assertRaises(AssertionError):
            task.run(template, science, sources)

    def test_mismatched_filter(self):
        """Test that an error is raised if the science and template have
        different bands.
        """
        xSize = 200
        ySize = 200
        science, sources = makeTestImage(psfSize=2.4, xSize=xSize + 20, ySize=ySize + 20,
                                         band="g", physicalFilter="g noCamera")
        template, _ = makeTestImage(psfSize=2.4, xSize=xSize, ySize=ySize, doApplyCalibration=True,
                                    band="not-g", physicalFilter="not-g noCamera")
        task = self._setup_subtraction()
        with self.assertRaises(AssertionError):
            task.run(template, science, sources)

    def test_incomplete_template_coverage(self):
        noiseLevel = 1.
        border = 20
        xSize = 400
        ySize = 400
        science, sources = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel, noiseSeed=6, nSrc=50,
                                         xSize=xSize, ySize=ySize)
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=7, nSrc=50,
                                    templateBorderSize=border, doApplyCalibration=True,
                                    xSize=xSize, ySize=ySize)

        science_height = science.getBBox().getDimensions().getY()

        def _run_and_check_coverage(template_coverage,
                                    requiredTemplateFraction=0.1,
                                    minTemplateFractionForExpectedSuccess=0.2):
            template_cut = template.clone()
            template_height = int(science_height*template_coverage + border)
            template_cut.image.array[:, template_height:] = 0
            template_cut.mask.array[:, template_height:] = template_cut.mask.getPlaneBitMask('NO_DATA')
            task = self._setup_subtraction(
                requiredTemplateFraction=requiredTemplateFraction,
                minTemplateFractionForExpectedSuccess=minTemplateFractionForExpectedSuccess
            )
            if template_coverage < requiredTemplateFraction:
                doRaise = True
            elif template_coverage < minTemplateFractionForExpectedSuccess:
                doRaise = True
            else:
                doRaise = False
            if doRaise:
                with self.assertRaises(NoWorkFound):
                    task.run(template_cut, science.clone(), sources.copy(deep=True))
            else:
                task.run(template_cut, science.clone(), sources.copy(deep=True))
        _run_and_check_coverage(template_coverage=0.09)
        _run_and_check_coverage(template_coverage=0.19)
        _run_and_check_coverage(template_coverage=0.7)

    def test_clear_template_mask(self):
        noiseLevel = 1.
        science, sources = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel, noiseSeed=6)
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True)
        diffimEmptyMaskPlanes = ["DETECTED", "DETECTED_NEGATIVE"]
        task = self._setup_subtraction(mode="convolveTemplate")
        # Ensure that each each mask plane is set for some pixels
        mask = template.mask
        x0 = 50
        x1 = 75
        y0 = 150
        y1 = 175
        scienceMaskCheck = {}
        for maskPlane in mask.getMaskPlaneDict().keys():
            scienceMaskCheck[maskPlane] = np.sum(science.mask.array & mask.getPlaneBitMask(maskPlane) > 0)
            mask.array[x0: x1, y0: y1] |= mask.getPlaneBitMask(maskPlane)
            self.assertTrue(np.sum(mask.array & mask.getPlaneBitMask(maskPlane) > 0))

        output = task.run(template, science, sources)
        # Verify that the template mask has been modified in place
        for maskPlane in mask.getMaskPlaneDict().keys():
            if maskPlane in diffimEmptyMaskPlanes:
                self.assertTrue(np.sum(mask.array & mask.getPlaneBitMask(maskPlane) == 0))
            elif maskPlane in task.config.preserveTemplateMask:
                self.assertTrue(np.sum(mask.array & mask.getPlaneBitMask(maskPlane) > 0))
            else:
                self.assertTrue(np.sum(mask.array & mask.getPlaneBitMask(maskPlane) == 0))
        # Mask planes set in the science image should also be set in the difference
        # Except the "DETECTED" planes should have been cleared
        diffimMask = output.difference.mask
        for maskPlane, scienceSum in scienceMaskCheck.items():
            diffimSum = np.sum(diffimMask.array & mask.getPlaneBitMask(maskPlane) > 0)
            if maskPlane in diffimEmptyMaskPlanes:
                self.assertEqual(diffimSum, 0)
            else:
                self.assertTrue(diffimSum >= scienceSum)

    def test_equal_images(self):
        """Test that running with enough sources produces reasonable output,
        with the same size psf in the template and science.
        """
        noiseLevel = 1.
        science, sources = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel, noiseSeed=6)
        template, _ = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True)
        task = self._setup_subtraction()
        output = task.run(template, science, sources)
        # There shoud be no NaN values in the difference image
        self.assertTrue(np.all(np.isfinite(output.difference.image.array)))
        # Mean of difference image should be close to zero.
        meanError = noiseLevel/np.sqrt(output.difference.image.array.size)
        # Make sure to include pixels with the DETECTED mask bit set.
        statsCtrl = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA", "DETECTED", "DETECTED_NEGATIVE"))
        differenceMean = computeRobustStatistics(output.difference.image, output.difference.mask, statsCtrl)
        self.assertFloatsAlmostEqual(differenceMean, 0, atol=5*meanError)
        # stddev of difference image should be close to expected value.
        differenceStd = computeRobustStatistics(output.difference.image, output.difference.mask,
                                                makeStats(), statistic=afwMath.STDEV)
        self.assertFloatsAlmostEqual(differenceStd, np.sqrt(2)*noiseLevel, rtol=0.1)

    def test_equal_images_missing_mask_planes(self):
        """Test that running with enough sources produces reasonable output,
        with the same size psf in the template and science and with missing
        mask planes.
        """
        noiseLevel = 1.
        science, sources = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel, noiseSeed=6, addMaskPlanes=[])
        template, _ = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True, addMaskPlanes=[])
        task = self._setup_subtraction()
        output = task.run(template, science, sources)
        # There shoud be no NaN values in the difference image
        self.assertTrue(np.all(np.isfinite(output.difference.image.array)))
        # Mean of difference image should be close to zero.
        meanError = noiseLevel/np.sqrt(output.difference.image.array.size)
        # Make sure to include pixels with the DETECTED mask bit set.
        statsCtrl = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA", "DETECTED", "DETECTED_NEGATIVE"))
        differenceMean = computeRobustStatistics(output.difference.image, output.difference.mask, statsCtrl)
        self.assertFloatsAlmostEqual(differenceMean, 0, atol=5*meanError)
        # stddev of difference image should be close to expected value.
        differenceStd = computeRobustStatistics(output.difference.image, output.difference.mask,
                                                makeStats(), statistic=afwMath.STDEV)
        self.assertFloatsAlmostEqual(differenceStd, np.sqrt(2)*noiseLevel, rtol=0.1)

    def test_psf_size(self):
        """Test that the image subtract task runs without failing, if
        fwhmExposureBuffer and fwhmExposureGrid parameters are set.
        """
        noiseLevel = 1.
        science, sources = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel, noiseSeed=6)
        template, _ = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True)

        schema = afwTable.ExposureTable.makeMinimalSchema()
        weightKey = schema.addField("weight", type="D", doc="Coadd weight")
        exposureCatalog = afwTable.ExposureCatalog(schema)
        kernel = measAlg.DoubleGaussianPsf(7, 7, 2.0).getKernel()
        psf = measAlg.KernelPsf(kernel, template.getBBox().getCenter())

        record = exposureCatalog.addNew()
        record.setPsf(psf)
        record.setWcs(template.wcs)
        record.setD(weightKey, 1.0)
        record.setBBox(template.getBBox())

        customPsf = CustomCoaddPsf(exposureCatalog, template.wcs)
        template.setPsf(customPsf)

        # Test that we get an exception if we simply get the FWHM at center.
        with self.assertRaises(InvalidParameterError):
            getPsfFwhm(template.psf, True)

        with self.assertRaises(InvalidParameterError):
            getPsfFwhm(template.psf, False)

        # Test that evaluateMeanPsfFwhm runs successfully on the template.
        evaluateMeanPsfFwhm(template, fwhmExposureBuffer=0.05, fwhmExposureGrid=10)

        # Since the PSF is spatially invariant, the FWHM should be the same at
        # all points in the science image.
        fwhm1 = getPsfFwhm(science.psf, False)
        fwhm2 = evaluateMeanPsfFwhm(science, fwhmExposureBuffer=0.05, fwhmExposureGrid=10)
        self.assertAlmostEqual(fwhm1[0], fwhm2, places=13)
        self.assertAlmostEqual(fwhm1[1], fwhm2, places=13)

        self.assertAlmostEqual(evaluateMeanPsfFwhm(science, fwhmExposureBuffer=0.05,
                                                   fwhmExposureGrid=10),
                               getPsfFwhm(science.psf, True), places=7
                               )

        # Test that the image subtraction task runs successfully.
        task = self._setup_subtraction()

        # Test that the task runs if we take the mean FWHM on a grid.
        with self.assertLogs(level="INFO") as cm:
            task.run(template, science, sources)

        # Check that evaluateMeanPsfFwhm was called.
        # This tests that getPsfFwhm failed raising InvalidParameterError,
        # that is caught and handled appropriately.
        logMessage = ("INFO:lsst.alardLuptonSubtract:Unable to evaluate PSF at the average position. "
                      "Evaluting PSF on a grid of points."
                      )
        self.assertIn(logMessage, cm.output)

    def test_auto_convolveTemplate(self):
        """Test that auto mode gives the same result as convolveTemplate when
        the template psf is the smaller.
        """
        noiseLevel = 1.
        science, sources = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel, noiseSeed=6)
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True)
        task = self._setup_subtraction(mode="convolveTemplate")
        output = task.run(template.clone(), science.clone(), sources)

        task = self._setup_subtraction(mode="auto")
        outputAuto = task.run(template, science, sources)
        self.assertMaskedImagesEqual(output.difference.maskedImage, outputAuto.difference.maskedImage)

    def test_auto_convolveScience(self):
        """Test that auto mode gives the same result as convolveScience when
        the science psf is the smaller.
        """
        noiseLevel = 1.
        science, sources = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=6)
        template, _ = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True)
        task = self._setup_subtraction(mode="convolveScience")
        output = task.run(template.clone(), science.clone(), sources)

        task = self._setup_subtraction(mode="auto")
        outputAuto = task.run(template, science, sources)
        self.assertMaskedImagesEqual(output.difference.maskedImage, outputAuto.difference.maskedImage)

    def test_science_better(self):
        """Test that running with enough sources produces reasonable output,
        with the science psf being smaller than the template.
        """
        statsCtrl = makeStats()
        statsCtrlDetect = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA"))

        def _run_and_check_images(statsCtrl, statsCtrlDetect, scienceNoiseLevel, templateNoiseLevel):
            science, sources = makeTestImage(psfSize=2.0, noiseLevel=scienceNoiseLevel, noiseSeed=6)
            template, _ = makeTestImage(psfSize=3.0, noiseLevel=templateNoiseLevel, noiseSeed=7,
                                        templateBorderSize=20, doApplyCalibration=True)
            task = self._setup_subtraction(mode="convolveScience")
            output = task.run(template, science, sources)
            self.assertFloatsAlmostEqual(task.metadata["scaleTemplateVarianceFactor"], 1., atol=.05)
            self.assertFloatsAlmostEqual(task.metadata["scaleScienceVarianceFactor"], 1., atol=.05)
            # Mean of difference image should be close to zero.
            nGoodPix = np.sum(np.isfinite(output.difference.image.array))
            meanError = (scienceNoiseLevel + templateNoiseLevel)/np.sqrt(nGoodPix)
            diffimMean = computeRobustStatistics(output.difference.image, output.difference.mask,
                                                 statsCtrlDetect)

            self.assertFloatsAlmostEqual(diffimMean, 0, atol=5*meanError)
            # stddev of difference image should be close to expected value.
            noiseLevel = np.sqrt(scienceNoiseLevel**2 + templateNoiseLevel**2)
            varianceMean = computeRobustStatistics(output.difference.variance, output.difference.mask,
                                                   statsCtrl)
            diffimStd = computeRobustStatistics(output.difference.image, output.difference.mask,
                                                statsCtrl, statistic=afwMath.STDEV)
            self.assertFloatsAlmostEqual(varianceMean, noiseLevel**2, rtol=0.1)
            self.assertFloatsAlmostEqual(diffimStd, noiseLevel, rtol=0.1)

        _run_and_check_images(statsCtrl, statsCtrlDetect, scienceNoiseLevel=1., templateNoiseLevel=1.)
        _run_and_check_images(statsCtrl, statsCtrlDetect, scienceNoiseLevel=1., templateNoiseLevel=.1)
        _run_and_check_images(statsCtrl, statsCtrlDetect, scienceNoiseLevel=.1, templateNoiseLevel=.1)

    def test_template_better(self):
        """Test that running with enough sources produces reasonable output,
        with the template psf being smaller than the science.
        """
        statsCtrl = makeStats()
        statsCtrlDetect = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA"))

        def _run_and_check_images(statsCtrl, statsCtrlDetect, scienceNoiseLevel, templateNoiseLevel):
            science, sources = makeTestImage(psfSize=3.0, noiseLevel=scienceNoiseLevel, noiseSeed=6)
            template, _ = makeTestImage(psfSize=2.0, noiseLevel=templateNoiseLevel, noiseSeed=7,
                                        templateBorderSize=20, doApplyCalibration=True)
            task = self._setup_subtraction()
            output = task.run(template, science, sources)
            self.assertFloatsAlmostEqual(task.metadata["scaleTemplateVarianceFactor"], 1., atol=.05)
            self.assertFloatsAlmostEqual(task.metadata["scaleScienceVarianceFactor"], 1., atol=.05)
            # There should be no NaNs in the image if we convolve the template with a buffer
            self.assertTrue(np.all(np.isfinite(output.difference.image.array)))
            # Mean of difference image should be close to zero.
            meanError = (scienceNoiseLevel + templateNoiseLevel)/np.sqrt(output.difference.image.array.size)

            diffimMean = computeRobustStatistics(output.difference.image, output.difference.mask,
                                                 statsCtrlDetect)
            self.assertFloatsAlmostEqual(diffimMean, 0, atol=5*meanError)
            # stddev of difference image should be close to expected value.
            noiseLevel = np.sqrt(scienceNoiseLevel**2 + templateNoiseLevel**2)
            varianceMean = computeRobustStatistics(output.difference.variance, output.difference.mask,
                                                   statsCtrl)
            diffimStd = computeRobustStatistics(output.difference.image, output.difference.mask,
                                                statsCtrl, statistic=afwMath.STDEV)
            self.assertFloatsAlmostEqual(varianceMean, noiseLevel**2, rtol=0.1)
            self.assertFloatsAlmostEqual(diffimStd, noiseLevel, rtol=0.1)

        _run_and_check_images(statsCtrl, statsCtrlDetect, scienceNoiseLevel=1., templateNoiseLevel=1.)
        _run_and_check_images(statsCtrl, statsCtrlDetect, scienceNoiseLevel=1., templateNoiseLevel=.1)
        _run_and_check_images(statsCtrl, statsCtrlDetect, scienceNoiseLevel=.1, templateNoiseLevel=.1)

    def test_symmetry(self):
        """Test that convolving the science and convolving the template are
        symmetric: if the psfs are switched between them, the difference image
        should be nearly the same.
        """
        noiseLevel = 1.
        # Don't include a border for the template, in order to make the results
        #  comparable when we swap which image is treated as the "science" image.
        science, sources = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel,
                                         noiseSeed=6, templateBorderSize=0, doApplyCalibration=True)
        template, _ = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel,
                                    noiseSeed=7, templateBorderSize=0, doApplyCalibration=True)
        task = self._setup_subtraction(mode='auto')

        # The science image will be modified in place, so use a copy for the second run.
        science_better = task.run(template.clone(), science.clone(), sources)
        template_better = task.run(science, template, sources)

        delta = template_better.difference.clone()
        delta.image -= science_better.difference.image
        delta.variance -= science_better.difference.variance
        delta.mask.array -= science_better.difference.mask.array

        statsCtrl = makeStats()
        # Mean of delta should be very close to zero.
        nGoodPix = np.sum(np.isfinite(delta.image.array))
        meanError = 2*noiseLevel/np.sqrt(nGoodPix)
        deltaMean = computeRobustStatistics(delta.image, delta.mask, statsCtrl)
        deltaStd = computeRobustStatistics(delta.image, delta.mask, statsCtrl, statistic=afwMath.STDEV)
        self.assertFloatsAlmostEqual(deltaMean, 0, atol=5*meanError)
        # stddev of difference image should be close to expected value
        self.assertFloatsAlmostEqual(deltaStd, 2*np.sqrt(2)*noiseLevel, rtol=.1)

    def test_few_sources(self):
        """Test with only 1 source, to check that we get a useful error.
        """
        xSize = 256
        ySize = 256
        science, sources = makeTestImage(psfSize=2.4, nSrc=10, xSize=xSize, ySize=ySize)
        template, _ = makeTestImage(psfSize=2.0, nSrc=10, xSize=xSize, ySize=ySize, doApplyCalibration=True)
        task = self._setup_subtraction()
        sources = sources[0:1]
        with self.assertRaisesRegex(RuntimeError,
                                    "Cannot compute PSF matching kernel: too few sources selected."):
            task.run(template, science, sources)

    def test_kernel_source_selector(self):
        """Check that kernel source selection behaves as expected.
        """
        xSize = 256
        ySize = 256
        nSourcesSimulated = 20
        science, sources = makeTestImage(psfSize=2.4, nSrc=nSourcesSimulated,
                                         xSize=xSize, ySize=ySize)
        template, _ = makeTestImage(psfSize=2.0, nSrc=nSourcesSimulated,
                                    xSize=xSize, ySize=ySize, doApplyCalibration=True)

        def _run_and_check_sources(sourcesIn, maxKernelSources=1000, minKernelSources=3):
            sources = sourcesIn.copy(deep=True)

            task = self._setup_subtraction(maxKernelSources=maxKernelSources,
                                           minKernelSources=minKernelSources,
                                           )
            # Verify that source flags are not set in the input catalog
            # Note that this will use the last flag in the list for the rest of
            #  the test.
            for badSourceFlag in task.sourceSelector.config.flags.bad:
                self.assertEqual(np.sum(sources[badSourceFlag]), 0)
            nSources = len(sources)
            # Flag a third of the sources
            sources[0:: 3][badSourceFlag] = True
            nBadSources = np.sum(sources[badSourceFlag])
            if maxKernelSources > 0:
                nGoodSources = np.minimum(nSources - nBadSources, maxKernelSources)
            else:
                nGoodSources = nSources - nBadSources

            signalToNoise = sources.getPsfInstFlux()/sources.getPsfInstFluxErr()
            signalToNoise = signalToNoise[~sources[badSourceFlag]]
            signalToNoise.sort()
            selectSources = task._sourceSelector(sources, science.mask)
            self.assertEqual(nGoodSources, len(selectSources))
            signalToNoiseOut = selectSources.getPsfInstFlux()/selectSources.getPsfInstFluxErr()
            signalToNoiseOut.sort()
            self.assertFloatsAlmostEqual(signalToNoise[-nGoodSources:], signalToNoiseOut)

        _run_and_check_sources(sources)
        _run_and_check_sources(sources, maxKernelSources=len(sources)//3)
        _run_and_check_sources(sources, maxKernelSources=-1)
        with self.assertRaises(RuntimeError):
            _run_and_check_sources(sources, minKernelSources=1000)

    def test_order_equal_images(self):
        """Verify that the result is the same regardless of convolution mode
        if the images are equivalent.
        """
        noiseLevel = .1
        seed1 = 6
        seed2 = 7
        science1, sources1 = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=seed1,
                                           clearEdgeMask=True)
        template1, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=seed2,
                                     templateBorderSize=0, doApplyCalibration=True,
                                     clearEdgeMask=True)
        task1 = self._setup_subtraction(mode="convolveTemplate")
        results_convolveTemplate = task1.run(template1, science1, sources1)

        science2, sources2 = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=seed1,
                                           clearEdgeMask=True)
        template2, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=seed2,
                                     templateBorderSize=0, doApplyCalibration=True,
                                     clearEdgeMask=True)
        task2 = self._setup_subtraction(mode="convolveScience")
        results_convolveScience = task2.run(template2, science2, sources2)
        bbox = results_convolveTemplate.difference.getBBox().clippedTo(
            results_convolveScience.difference.getBBox())
        diff1 = science1.maskedImage.clone()[bbox]
        diff1 -= template1.maskedImage[bbox]
        diff2 = science2.maskedImage.clone()[bbox]
        diff2 -= template2.maskedImage[bbox]
        self.assertFloatsAlmostEqual(results_convolveTemplate.difference[bbox].image.array,
                                     diff1.image.array,
                                     atol=noiseLevel*5.)
        self.assertFloatsAlmostEqual(results_convolveScience.difference[bbox].image.array,
                                     diff2.image.array,
                                     atol=noiseLevel*5.)
        diffErr = noiseLevel*2
        self.assertMaskedImagesAlmostEqual(results_convolveTemplate.difference[bbox].maskedImage,
                                           results_convolveScience.difference[bbox].maskedImage,
                                           atol=diffErr*5.)

    def test_combined(self):
        """Test the combined AL+DF psf matching kernel.
        """
        noiseLevel = .1
        seed1 = 6
        seed2 = 7
        science1, sources1 = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=seed1,
                                           clearEdgeMask=True)
        template1, _ = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel, noiseSeed=seed2,
                                     templateBorderSize=0, doApplyCalibration=True,
                                     clearEdgeMask=True)
        config = self.subtractTask.ConfigClass()
        config.doSubtractBackground = False
        config.sourceSelector.signalToNoise.fluxField = "truth_instFlux"
        config.sourceSelector.signalToNoise.errField = "truth_instFluxErr"
        config.makeKernel.kernel.name = "combined"
        # config.makeKernel.kernel.name = "AL"
        task = self.subtractTask(config=config)
        lsst.utils.logging.trace_set_at("ip.diffim", 5)
        results = task.run(template1, science1, sources1)
        # TODO: need some asserts on the resulting diffim here!

    def test_background_subtraction(self):
        """Check that we can recover the background,
        and that it is subtracted correctly in the difference image.
        """
        noiseLevel = 1.
        xSize = 512
        ySize = 512
        x0 = 123
        y0 = 456
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20,
                                    xSize=xSize, ySize=ySize, x0=x0, y0=y0,
                                    doApplyCalibration=True)
        params = [2.2, 2.1, 2.0, 1.2, 1.1, 1.0]

        bbox2D = lsst.geom.Box2D(lsst.geom.Point2D(x0, y0), lsst.geom.Extent2D(xSize, ySize))
        background_model = afwMath.Chebyshev1Function2D(params, bbox2D)
        science, sources = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=6,
                                         background=background_model,
                                         xSize=xSize, ySize=ySize, x0=x0, y0=y0)
        # Don't use ``self._setup_subtraction()`` here.
        # Modifying the config of a subtask is messy.
        config = subtractImages.AlardLuptonSubtractTask.ConfigClass()

        config.sourceSelector.signalToNoise.fluxField = "truth_instFlux"
        config.sourceSelector.signalToNoise.errField = "truth_instFluxErr"
        config.doSubtractBackground = True

        config.makeKernel.kernel.name = "AL"
        config.makeKernel.kernel.active.fitForBackground = True
        config.makeKernel.kernel.active.spatialKernelOrder = 1
        config.makeKernel.kernel.active.spatialBgOrder = 2
        statsCtrl = makeStats()

        def _run_and_check_images(config, statsCtrl, mode):
            """Check that the fit background matches the input model.
            """
            config.mode = mode
            task = subtractImages.AlardLuptonSubtractTask(config=config)
            output = task.run(template.clone(), science.clone(), sources)

            # We should be fitting the same number of parameters as were in the input model
            self.assertEqual(output.backgroundModel.getNParameters(), background_model.getNParameters())

            # The parameters of the background fit should be close to the input model
            self.assertFloatsAlmostEqual(np.array(output.backgroundModel.getParameters()),
                                         np.array(params), rtol=0.3)

            # stddev of difference image should be close to expected value.
            # This will fail if we have mis-subtracted the background.
            stdVal = computeRobustStatistics(output.difference.image, output.difference.mask,
                                             statsCtrl, statistic=afwMath.STDEV)
            self.assertFloatsAlmostEqual(stdVal, np.sqrt(2)*noiseLevel, rtol=0.1)

        _run_and_check_images(config, statsCtrl, "convolveTemplate")
        _run_and_check_images(config, statsCtrl, "convolveScience")

    def test_scale_variance_convolve_template(self):
        """Check variance scaling of the image difference.
        """
        scienceNoiseLevel = 4.
        templateNoiseLevel = 2.
        scaleFactor = 1.345
        # Make sure to include pixels with the DETECTED mask bit set.
        statsCtrl = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA"))

        def _run_and_check_images(science, template, sources, statsCtrl,
                                  doDecorrelation, doScaleVariance, scaleFactor=1.):
            """Check that the variance plane matches the expected value for
            different configurations of ``doDecorrelation`` and ``doScaleVariance``.
            """

            task = self._setup_subtraction(doDecorrelation=doDecorrelation,
                                           doScaleVariance=doScaleVariance,
                                           )
            output = task.run(template.clone(), science.clone(), sources)
            if doScaleVariance:
                self.assertFloatsAlmostEqual(task.metadata["scaleTemplateVarianceFactor"],
                                             scaleFactor, atol=0.05)
                self.assertFloatsAlmostEqual(task.metadata["scaleScienceVarianceFactor"],
                                             scaleFactor, atol=0.05)

            scienceNoise = computeRobustStatistics(science.variance, science.mask, statsCtrl)
            if doDecorrelation:
                templateNoise = computeRobustStatistics(template.variance, template.mask, statsCtrl)
            else:
                templateNoise = computeRobustStatistics(output.matchedTemplate.variance,
                                                        output.matchedTemplate.mask,
                                                        statsCtrl)

            if doScaleVariance:
                templateNoise *= scaleFactor
                scienceNoise *= scaleFactor
            varMean = computeRobustStatistics(output.difference.variance, output.difference.mask, statsCtrl)
            self.assertFloatsAlmostEqual(varMean, scienceNoise + templateNoise, rtol=0.1)

        science, sources = makeTestImage(psfSize=3.0, noiseLevel=scienceNoiseLevel, noiseSeed=6)
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=templateNoiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True)
        # Verify that the variance plane of the difference image is correct
        #  when the template and science variance planes are correct
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=True)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=False)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=True)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=False)

        # Verify that the variance plane of the difference image is correct
        #  when the template variance plane is incorrect
        template.variance.array /= scaleFactor
        science.variance.array /= scaleFactor
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=True, scaleFactor=scaleFactor)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=False, scaleFactor=scaleFactor)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=True, scaleFactor=scaleFactor)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=False, scaleFactor=scaleFactor)

    def test_scale_variance_convolve_science(self):
        """Check variance scaling of the image difference.
        """
        scienceNoiseLevel = 4.
        templateNoiseLevel = 2.
        scaleFactor = 1.345
        # Make sure to include pixels with the DETECTED mask bit set.
        statsCtrl = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA"))

        def _run_and_check_images(science, template, sources, statsCtrl,
                                  doDecorrelation, doScaleVariance, scaleFactor=1.):
            """Check that the variance plane matches the expected value for
            different configurations of ``doDecorrelation`` and ``doScaleVariance``.
            """

            task = self._setup_subtraction(mode="convolveScience",
                                           doDecorrelation=doDecorrelation,
                                           doScaleVariance=doScaleVariance,
                                           )
            output = task.run(template.clone(), science.clone(), sources)
            if doScaleVariance:
                self.assertFloatsAlmostEqual(task.metadata["scaleTemplateVarianceFactor"],
                                             scaleFactor, atol=0.05)
                self.assertFloatsAlmostEqual(task.metadata["scaleScienceVarianceFactor"],
                                             scaleFactor, atol=0.05)

            templateNoise = computeRobustStatistics(template.variance, template.mask, statsCtrl)
            if doDecorrelation:
                scienceNoise = computeRobustStatistics(science.variance, science.mask, statsCtrl)
            else:
                scienceNoise = computeRobustStatistics(output.matchedScience.variance,
                                                       output.matchedScience.mask,
                                                       statsCtrl)

            if doScaleVariance:
                templateNoise *= scaleFactor
                scienceNoise *= scaleFactor

            varMean = computeRobustStatistics(output.difference.variance, output.difference.mask, statsCtrl)
            self.assertFloatsAlmostEqual(varMean, scienceNoise + templateNoise, rtol=0.1)

        science, sources = makeTestImage(psfSize=2.0, noiseLevel=scienceNoiseLevel, noiseSeed=6)
        template, _ = makeTestImage(psfSize=3.0, noiseLevel=templateNoiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True)
        # Verify that the variance plane of the difference image is correct
        #  when the template and science variance planes are correct
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=True)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=False)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=True)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=False)

        # Verify that the variance plane of the difference image is correct
        #  when the template and science variance planes are incorrect
        science.variance.array /= scaleFactor
        template.variance.array /= scaleFactor
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=True, scaleFactor=scaleFactor)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=False, scaleFactor=scaleFactor)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=True, scaleFactor=scaleFactor)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=False, scaleFactor=scaleFactor)

    def test_exposure_properties_convolve_template(self):
        """Check that all necessary exposure metadata is included
        when the template is convolved.
        """
        noiseLevel = 1.
        seed = 37
        rng = np.random.RandomState(seed)
        science, sources = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel, noiseSeed=6)
        psf = science.psf
        psfAvgPos = psf.getAveragePosition()
        psfSize = getPsfFwhm(science.psf)
        psfImg = psf.computeKernelImage(psfAvgPos)
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True)

        # Generate a random aperture correction map
        apCorrMap = lsst.afw.image.ApCorrMap()
        for name in ("a", "b", "c"):
            apCorrMap.set(name, lsst.afw.math.ChebyshevBoundedField(science.getBBox(), rng.randn(3, 3)))
        science.info.setApCorrMap(apCorrMap)

        def _run_and_check_images(doDecorrelation):
            """Check that the metadata is correct with or without decorrelation.
            """
            task = self._setup_subtraction(mode="convolveTemplate",
                                           doDecorrelation=doDecorrelation,
                                           )
            output = task.run(template.clone(), science.clone(), sources)
            psfOut = output.difference.psf
            psfAvgPos = psfOut.getAveragePosition()
            if doDecorrelation:
                # Decorrelation requires recalculating the PSF,
                #  so it will not be the same as the input
                psfOutSize = getPsfFwhm(science.psf)
                self.assertFloatsAlmostEqual(psfSize, psfOutSize)
            else:
                psfOutImg = psfOut.computeKernelImage(psfAvgPos)
                self.assertImagesAlmostEqual(psfImg, psfOutImg)

            # check PSF, WCS, bbox, filterLabel, photoCalib, aperture correction
            self._compare_apCorrMaps(apCorrMap, output.difference.info.getApCorrMap())
            self.assertWcsAlmostEqualOverBBox(science.wcs, output.difference.wcs, science.getBBox())
            self.assertEqual(science.filter, output.difference.filter)
            self.assertEqual(science.photoCalib, output.difference.photoCalib)
        _run_and_check_images(doDecorrelation=True)
        _run_and_check_images(doDecorrelation=False)

    def test_exposure_properties_convolve_science(self):
        """Check that all necessary exposure metadata is included
        when the science image is convolved.
        """
        noiseLevel = 1.
        seed = 37
        rng = np.random.RandomState(seed)
        science, sources = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=6)
        template, _ = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True)
        psf = template.psf
        psfAvgPos = psf.getAveragePosition()
        psfSize = getPsfFwhm(template.psf)
        psfImg = psf.computeKernelImage(psfAvgPos)

        # Generate a random aperture correction map
        apCorrMap = lsst.afw.image.ApCorrMap()
        for name in ("a", "b", "c"):
            apCorrMap.set(name, lsst.afw.math.ChebyshevBoundedField(science.getBBox(), rng.randn(3, 3)))
        science.info.setApCorrMap(apCorrMap)

        def _run_and_check_images(doDecorrelation):
            """Check that the metadata is correct with or without decorrelation.
            """
            task = self._setup_subtraction(mode="convolveScience",
                                           doDecorrelation=doDecorrelation,
                                           )
            output = task.run(template.clone(), science.clone(), sources)
            if doDecorrelation:
                # Decorrelation requires recalculating the PSF,
                #  so it will not be the same as the input
                psfOutSize = getPsfFwhm(template.psf)
                self.assertFloatsAlmostEqual(psfSize, psfOutSize)
            else:
                psfOut = output.difference.psf
                psfAvgPos = psfOut.getAveragePosition()
                psfOutImg = psfOut.computeKernelImage(psfAvgPos)
                self.assertImagesAlmostEqual(psfImg, psfOutImg)

            # check PSF, WCS, bbox, filterLabel, photoCalib, aperture correction
            self._compare_apCorrMaps(apCorrMap, output.difference.info.getApCorrMap())
            self.assertWcsAlmostEqualOverBBox(science.wcs, output.difference.wcs, science.getBBox())
            self.assertEqual(science.filter, output.difference.filter)
            self.assertEqual(science.photoCalib, output.difference.photoCalib)

        _run_and_check_images(doDecorrelation=True)
        _run_and_check_images(doDecorrelation=False)

    def _compare_apCorrMaps(self, a, b):
        """Compare two ApCorrMaps for equality, without assuming that their BoundedFields have the
        same addresses (i.e. so we can compare after serialization).

        This function is taken from ``ApCorrMapTestCase`` in afw/tests/.

        Parameters
        ----------
        a, b : `lsst.afw.image.ApCorrMap`
            The two aperture correction maps to compare.
        """
        self.assertEqual(len(a), len(b))
        for name, value in list(a.items()):
            value2 = b.get(name)
            self.assertIsNotNone(value2)
            self.assertEqual(value.getBBox(), value2.getBBox())
            self.assertFloatsAlmostEqual(
                value.getCoefficients(), value2.getCoefficients(), rtol=0.0)

    def test_fake_mask_plane_propagation(self):
        """Test that we have the mask planes related to fakes in diffim images.
        This is testing method called updateMasks
        """
        xSize = 200
        ySize = 200
        science, sources = makeTestImage(psfSize=2.4, xSize=xSize, ySize=ySize)
        science_fake_img, science_fake_sources = makeTestImage(
            psfSize=2.4, xSize=xSize, ySize=ySize, seed=7, nSrc=2, noiseLevel=0.25, fluxRange=1
        )
        template, _ = makeTestImage(psfSize=2.4, xSize=xSize, ySize=ySize, doApplyCalibration=True)
        tmplt_fake_img, tmplt_fake_sources = makeTestImage(
            psfSize=2.4, xSize=xSize, ySize=ySize, seed=9, nSrc=2, noiseLevel=0.25, fluxRange=1
        )
        # created fakes and added them to the images
        science.image.array += science_fake_img.image.array
        template.image.array += tmplt_fake_img.image.array

        # TODO: DM-40796 update to INJECTED names when source injection gets refactored
        # adding mask planes to both science and template images
        science_mask_planes = science.mask.addMaskPlane("FAKE")
        template_mask_planes = template.mask.addMaskPlane("FAKE")

        for a_science_source in science_fake_sources:
            # 3 x 3 masking of the source locations is fine
            bbox = lsst.geom.Box2I(
                lsst.geom.Point2I(a_science_source.getX(), a_science_source.getY()), lsst.geom.Extent2I(3, 3)
            )
            science[bbox].mask.array |= science_mask_planes

        for a_template_source in tmplt_fake_sources:
            # 3 x 3 masking of the source locations is fine
            bbox = lsst.geom.Box2I(
                lsst.geom.Point2I(a_template_source.getX(), a_template_source.getY()),
                lsst.geom.Extent2I(3, 3)
            )
            template[bbox].mask.array |= template_mask_planes

        science_fake_masked = (science.mask.array & science.mask.getPlaneBitMask("FAKE")) > 0
        template_fake_masked = (template.mask.array & template.mask.getPlaneBitMask("FAKE")) > 0

        task = self._setup_subtraction()
        subtraction = task.run(template, science, sources)

        # check subtraction mask plane is set where we set the previous masks
        diff_mask = subtraction.difference.mask

        # science mask should be now in INJECTED
        inj_masked = (diff_mask.array & diff_mask.getPlaneBitMask("INJECTED")) > 0

        # template mask should be now in INJECTED_TEMPLATE
        injTmplt_masked = (diff_mask.array & diff_mask.getPlaneBitMask("INJECTED_TEMPLATE")) > 0

        self.assertEqual(np.sum(inj_masked.astype(int)-science_fake_masked.astype(int)), 0)
        self.assertEqual(np.sum(injTmplt_masked.astype(int)-template_fake_masked.astype(int)), 0)

    def test_metadata_metrics(self):
        """Verify fields are added to metadata when subtraction is run, and
        that the difference image limiting magnitude is calculated correctly,
        both with a "good" and "bad" seeing template.
        """
        science, sources = makeTestImage(psfSize=1.8, doApplyCalibration=True)
        template_good, _ = makeTestImage(psfSize=2.4, doApplyCalibration=True)
        template_bad, _ = makeTestImage(psfSize=9.5, doApplyCalibration=True)

        # Add a few sky objects; sky footprints are needed for some metrics.
        config = measAlg.SkyObjectsTask.ConfigClass()
        config.nSources = 3
        skyTask = measAlg.SkyObjectsTask(config=config, name="skySources")
        skyTask.skySourceKey = sources.schema["sky_source"].asKey()
        skyTask.run(science.mask, 10, catalog=sources)
        sources = sources.copy(deep=True)
        # Add centroids, since these sources were added post-measurement.
        for record in sources[sources["sky_source"]]:
            record["truth_x"] = record.getFootprint().getPeaks()[0].getFx()
            record["truth_y"] = record.getFootprint().getPeaks()[0].getFy()

        # The metadata fields are attached to the subtractTask, so we do
        # need to run that; run it for both "good" and "bad" seeing templates

        subtractTask_good = self._setup_subtraction()
        _ = subtractTask_good.run(template_good.clone(), science.clone(), sources)
        subtractTask_bad = self._setup_subtraction()
        _ = subtractTask_bad.run(template_bad.clone(), science.clone(), sources)

        # Test that the diffim limiting magnitudes are computed correctly
        maglim_science = subtractTask_good._calculateMagLim(science)
        fluxlim_science = (maglim_science*u.ABmag).to_value(u.nJy)
        maglim_template_good = subtractTask_good._calculateMagLim(template_good)
        fluxlim_template_good = (maglim_template_good*u.ABmag).to_value(u.nJy)
        maglim_template_bad = subtractTask_bad._calculateMagLim(template_bad)
        fluxlim_template_bad = (maglim_template_bad*u.ABmag).to_value(u.nJy)

        maglim_good = (np.sqrt(fluxlim_science**2 + fluxlim_template_good**2)*u.nJy).to(u.ABmag).value
        maglim_bad = (np.sqrt(fluxlim_science**2 + fluxlim_template_bad**2)*u.nJy).to(u.ABmag).value

        self.assertFloatsAlmostEqual(subtractTask_good.metadata['diffimLimitingMagnitude'],
                                     maglim_good, atol=1e-6)
        self.assertFloatsAlmostEqual(subtractTask_bad.metadata['diffimLimitingMagnitude'],
                                     maglim_bad, atol=1e-6)

        # Create a template with a PSF that is not defined at the image center.
        # First, make an exposure catalog so we can force the template to have
        # a bad (off-image) PSF. It must have a record with a weight field
        # and a BBox in order to let us set the PSF manually.
        template_offimage, _ = makeTestImage()
        schema = afwTable.ExposureTable.makeMinimalSchema()
        weightKey = schema.addField("weight", type="D", doc="Coadd weight")
        exposureCatalog = afwTable.ExposureCatalog(schema)
        record = exposureCatalog.addNew()
        record.setD(weightKey, 1.0)
        record.setBBox(template_offimage.getBBox())
        kernel = measAlg.DoubleGaussianPsf(7, 7, 2.0).getKernel()
        psf = measAlg.KernelPsf(kernel, template_offimage.getBBox().getCenter())
        record.setPsf(psf)
        record.setWcs(template_offimage.wcs)
        custom_offimage_psf = CustomCoaddPsf(exposureCatalog, template_offimage.wcs)
        template_offimage.setPsf(custom_offimage_psf)

        # Test that template PSF size edge cases are handled correctly.
        subtractTask_offimage = self._setup_subtraction()
        _ = subtractTask_offimage.run(template_offimage.clone(), science.clone(), sources)
        # Test that providing no fallbackPsfSize results in a nan template
        # limiting magnitude.
        maglim_template_offimage = subtractTask_offimage._calculateMagLim(template_offimage)
        self.assertTrue(np.isnan(maglim_template_offimage))
        # Test that given the provided fallbackPsfSize, the diffim limiting
        # magnitude is calculated correctly.
        maglim_template_offimage = 28.3173123103493
        fluxlim_template_offimage = (maglim_template_offimage*u.ABmag).to_value(u.nJy)
        maglim_offimage = (np.sqrt(fluxlim_science**2 + fluxlim_template_offimage**2)*u.nJy).to(u.ABmag).value
        self.assertEqual(subtractTask_offimage.metadata['diffimLimitingMagnitude'], maglim_offimage)

        # Test that several other expected metadata metrics exist
        self.assertIn('scienceLimitingMagnitude', subtractTask_good.metadata)
        self.assertIn('templateLimitingMagnitude', subtractTask_good.metadata)

        # The mean ratio metric should be much worse on the "bad" subtraction.
        self.assertLess(subtractTask_good.metadata['differenceFootprintRatioMean'], 0.2)
        self.assertGreater(subtractTask_bad.metadata['differenceFootprintRatioMean'], 1.0)


class AlardLuptonPreconvolveSubtractTest(AlardLuptonSubtractTestBase, lsst.utils.tests.TestCase):
    subtractTask = subtractImages.AlardLuptonPreconvolveSubtractTask

    def test_mismatched_template(self):
        """Test that an error is raised if the template
        does not fully contain the science image.
        """
        xSize = 200
        ySize = 200
        science, sources = makeTestImage(psfSize=2.4, xSize=xSize + 20, ySize=ySize + 20)
        template, _ = makeTestImage(psfSize=2.4, xSize=xSize, ySize=ySize, doApplyCalibration=True)
        task = self._setup_subtraction()
        with self.assertRaises(AssertionError):
            task.run(template, science, sources)

    def test_equal_images(self):
        """Test that running with enough sources produces reasonable output,
        with the same size psf in the template and science.
        """
        noiseLevel = 1.
        xSize = 400
        ySize = 400
        science, sources = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel, noiseSeed=6,
                                         xSize=xSize, ySize=ySize)
        template, _ = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True,
                                    xSize=xSize, ySize=ySize)
        task = self._setup_subtraction()
        output = task.run(template, science, sources)
        # There shoud be no NaN values in the Score image
        self.assertTrue(np.all(np.isfinite(output.scoreExposure.image.array)))
        # Mean of Score image should be close to zero.
        meanError = noiseLevel/np.sqrt(output.scoreExposure.image.array.size)
        # Make sure to include pixels with the DETECTED mask bit set.
        statsCtrl = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA"))
        scoreMean = computeRobustStatistics(output.scoreExposure.image,
                                            output.scoreExposure.mask,
                                            statsCtrl)
        self.assertFloatsAlmostEqual(scoreMean, 0, atol=5*meanError)
        nea = computePSFNoiseEquivalentArea(science.psf)
        # stddev of Score image should be close to expected value.
        scoreStd = computeRobustStatistics(output.scoreExposure.image, output.scoreExposure.mask,
                                           statsCtrl=statsCtrl, statistic=afwMath.STDEV)
        self.assertFloatsAlmostEqual(scoreStd, np.sqrt(2)*noiseLevel/np.sqrt(nea), rtol=0.1)

    def test_incomplete_template_coverage(self):
        noiseLevel = 1.
        border = 20
        xSize = 400
        ySize = 400
        science, sources = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel, noiseSeed=6,
                                         xSize=xSize, ySize=ySize)
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=border, doApplyCalibration=True,
                                    xSize=xSize, ySize=ySize)

        science_height = science.getBBox().getDimensions().getY()

        def _run_and_check_coverage(template_coverage):
            template_cut = template.clone()
            template_height = int(science_height*template_coverage + border)
            template_cut.image.array[:, template_height:] = 0
            template_cut.mask.array[:, template_height:] = template_cut.mask.getPlaneBitMask('NO_DATA')
            task = self._setup_subtraction()
            if template_coverage < task.config.requiredTemplateFraction:
                doRaise = True
            elif template_coverage < task.config.minTemplateFractionForExpectedSuccess:
                doRaise = True
            else:
                doRaise = False
            if doRaise:
                with self.assertRaises(NoWorkFound):
                    task.run(template_cut, science.clone(), sources.copy(deep=True))
            else:
                task.run(template_cut, science.clone(), sources.copy(deep=True))
        _run_and_check_coverage(template_coverage=0.09)
        _run_and_check_coverage(template_coverage=0.19)
        _run_and_check_coverage(template_coverage=.7)

    def test_clear_template_mask(self):
        noiseLevel = 1.
        xSize = 400
        ySize = 400
        science, sources = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel, noiseSeed=6,
                                         xSize=xSize, ySize=ySize)
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True,
                                    xSize=xSize, ySize=ySize)
        diffimEmptyMaskPlanes = ["DETECTED", "DETECTED_NEGATIVE"]
        task = self._setup_subtraction()
        # Ensure that each each mask plane is set for some pixels
        mask = template.mask
        x0 = 50
        x1 = 75
        y0 = 150
        y1 = 175
        scienceMaskCheck = {}
        for maskPlane in mask.getMaskPlaneDict().keys():
            scienceMaskCheck[maskPlane] = np.sum(science.mask.array & mask.getPlaneBitMask(maskPlane) > 0)
            mask.array[x0: x1, y0: y1] |= mask.getPlaneBitMask(maskPlane)
            self.assertTrue(np.sum(mask.array & mask.getPlaneBitMask(maskPlane) > 0))

        output = task.run(template, science, sources)
        # Verify that the template mask has been modified in place
        for maskPlane in mask.getMaskPlaneDict().keys():
            if maskPlane in diffimEmptyMaskPlanes:
                self.assertTrue(np.sum(mask.array & mask.getPlaneBitMask(maskPlane) == 0))
            elif maskPlane in task.config.preserveTemplateMask:
                self.assertTrue(np.sum(mask.array & mask.getPlaneBitMask(maskPlane) > 0))
            else:
                self.assertTrue(np.sum(mask.array & mask.getPlaneBitMask(maskPlane) == 0))
        # Mask planes set in the science image should also be set in the difference
        # Except the "DETECTED" planes should have been cleared
        diffimMask = output.scoreExposure.mask
        for maskPlane, scienceSum in scienceMaskCheck.items():
            diffimSum = np.sum(diffimMask.array & mask.getPlaneBitMask(maskPlane) > 0)
            if maskPlane in diffimEmptyMaskPlanes:
                self.assertEqual(diffimSum, 0)
            else:
                self.assertTrue(diffimSum >= scienceSum)

    def test_agnostic_template_psf(self):
        """Test that the Score image is the same whether the template PSF is
        larger or smaller than the science image PSF.
        """
        noiseLevel = .3
        xSize = 400
        ySize = 400
        science, sources = makeTestImage(psfSize=2.4, noiseLevel=noiseLevel,
                                         noiseSeed=6, templateBorderSize=0,
                                         xSize=xSize, ySize=ySize)
        template1, _ = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel,
                                     noiseSeed=7, doApplyCalibration=True,
                                     xSize=xSize, ySize=ySize)
        template2, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel,
                                     noiseSeed=8, doApplyCalibration=True,
                                     xSize=xSize, ySize=ySize)
        task = self._setup_subtraction()

        science_better = task.run(template1, science.clone(), sources)
        template_better = task.run(template2, science, sources)
        bbox = science_better.scoreExposure.getBBox().clippedTo(template_better.scoreExposure.getBBox())

        delta = template_better.scoreExposure[bbox].clone()
        delta.image -= science_better.scoreExposure[bbox].image
        delta.variance -= science_better.scoreExposure[bbox].variance
        delta.mask.array &= science_better.scoreExposure[bbox].mask.array

        statsCtrl = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA"))
        # Mean of delta should be very close to zero.
        nGoodPix = np.sum(np.isfinite(delta.image.array))
        meanError = 2*noiseLevel/np.sqrt(nGoodPix)
        deltaMean = computeRobustStatistics(delta.image, delta.mask, statsCtrl)
        deltaStd = computeRobustStatistics(delta.image, delta.mask, statsCtrl,
                                           statistic=afwMath.STDEV)
        self.assertFloatsAlmostEqual(deltaMean, 0, atol=5*meanError)
        nea = computePSFNoiseEquivalentArea(science.psf)
        # stddev of Score image should be close to expected value
        self.assertFloatsAlmostEqual(deltaStd, np.sqrt(2)*noiseLevel/np.sqrt(nea), rtol=.1)

    def test_few_sources(self):
        """Test with only 1 source, to check that we get a useful error.
        """
        xSize = 256
        ySize = 256
        science, sources = makeTestImage(psfSize=2.4, nSrc=10, xSize=xSize, ySize=ySize)
        template, _ = makeTestImage(psfSize=2.0, nSrc=10, xSize=xSize, ySize=ySize, doApplyCalibration=True)
        task = self._setup_subtraction()
        sources = sources[0:1]
        with self.assertRaisesRegex(RuntimeError,
                                    "Cannot compute PSF matching kernel: too few sources selected."):
            task.run(template, science, sources)

    def test_background_subtraction(self):
        """Check that we can recover the background,
        and that it is subtracted correctly in the Score image.
        """
        noiseLevel = 1.
        xSize = 512
        ySize = 512
        x0 = 123
        y0 = 456
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20,
                                    xSize=xSize, ySize=ySize, x0=x0, y0=y0,
                                    doApplyCalibration=True)
        params = [2.2, 2.1, 2.0, 1.2, 1.1, 1.0]

        bbox2D = lsst.geom.Box2D(lsst.geom.Point2D(x0, y0), lsst.geom.Extent2D(xSize, ySize))
        background_model = afwMath.Chebyshev1Function2D(params, bbox2D)
        science, sources = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=6,
                                         background=background_model,
                                         xSize=xSize, ySize=ySize, x0=x0, y0=y0)
        # Don't use ``self._setup_subtraction()`` here.
        # Modifying the config of a subtask is messy.
        config = subtractImages.AlardLuptonPreconvolveSubtractTask.ConfigClass()

        config.sourceSelector.signalToNoise.fluxField = "truth_instFlux"
        config.sourceSelector.signalToNoise.errField = "truth_instFluxErr"
        config.doSubtractBackground = True

        config.makeKernel.kernel.name = "AL"
        config.makeKernel.kernel.active.fitForBackground = True
        config.makeKernel.kernel.active.spatialKernelOrder = 1
        config.makeKernel.kernel.active.spatialBgOrder = 2
        statsCtrl = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA"))

        task = subtractImages.AlardLuptonPreconvolveSubtractTask(config=config)
        output = task.run(template.clone(), science.clone(), sources)

        # We should be fitting the same number of parameters as were in the input model
        self.assertEqual(output.backgroundModel.getNParameters(), background_model.getNParameters())

        # The parameters of the background fit should be close to the input model
        self.assertFloatsAlmostEqual(np.array(output.backgroundModel.getParameters()),
                                     np.array(params), rtol=0.2)

        # stddev of Score image should be close to expected value.
        # This will fail if we have mis-subtracted the background.
        stdVal = computeRobustStatistics(output.scoreExposure.image, output.scoreExposure.mask,
                                         statsCtrl, statistic=afwMath.STDEV)
        # get the img psf Noise Equivalent Area value
        nea = computePSFNoiseEquivalentArea(science.psf)
        self.assertFloatsAlmostEqual(stdVal, np.sqrt(2)*noiseLevel/np.sqrt(nea), rtol=0.1)

    def test_scale_variance(self):
        """Check variance scaling of the Score image.
        """
        scienceNoiseLevel = 4.
        templateNoiseLevel = 2.
        scaleFactor = 1.345
        xSize = 400
        ySize = 400
        # Make sure to include pixels with the DETECTED mask bit set.
        statsCtrl = makeStats(badMaskPlanes=("EDGE", "BAD", "NO_DATA"))

        def _run_and_check_images(science, template, sources, statsCtrl,
                                  doDecorrelation, doScaleVariance, scaleFactor=1.):
            """Check that the variance plane matches the expected value for
            different configurations of ``doDecorrelation`` and ``doScaleVariance``.
            """

            task = self._setup_subtraction(doDecorrelation=doDecorrelation,
                                           doScaleVariance=doScaleVariance,
                                           )
            output = task.run(template.clone(), science.clone(), sources)
            if doScaleVariance:
                self.assertFloatsAlmostEqual(task.metadata["scaleTemplateVarianceFactor"],
                                             scaleFactor, atol=0.05)
                self.assertFloatsAlmostEqual(task.metadata["scaleScienceVarianceFactor"],
                                             scaleFactor, atol=0.05)

            scienceNoise = computeRobustStatistics(science.variance, science.mask, statsCtrl)
            # get the img psf Noise Equivalent Area value
            nea = computePSFNoiseEquivalentArea(science.psf)
            scienceNoise /= nea
            if doDecorrelation:
                templateNoise = computeRobustStatistics(template.variance, template.mask, statsCtrl)
                templateNoise /= nea
            else:
                # Don't divide by NEA in this case, since the template is convolved
                #  and in the same units as the Score exposure.
                templateNoise = computeRobustStatistics(output.matchedTemplate.variance,
                                                        output.matchedTemplate.mask,
                                                        statsCtrl)
            if doScaleVariance:
                templateNoise *= scaleFactor
                scienceNoise *= scaleFactor
            varMean = computeRobustStatistics(output.scoreExposure.variance,
                                              output.scoreExposure.mask,
                                              statsCtrl)
            self.assertFloatsAlmostEqual(varMean, scienceNoise + templateNoise, rtol=0.1)

        science, sources = makeTestImage(psfSize=3.0, noiseLevel=scienceNoiseLevel, noiseSeed=6,
                                         xSize=xSize, ySize=ySize)
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=templateNoiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True,
                                    xSize=xSize, ySize=ySize)
        # Verify that the variance plane of the Score image is correct
        #  when the template and science variance planes are correct
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=True)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=False)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=True)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=False)

        # Verify that the variance plane of the Score image is correct
        #  when the template variance plane is incorrect
        template.variance.array /= scaleFactor
        science.variance.array /= scaleFactor
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=True, scaleFactor=scaleFactor)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=True, doScaleVariance=False, scaleFactor=scaleFactor)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=True, scaleFactor=scaleFactor)
        _run_and_check_images(science, template, sources, statsCtrl,
                              doDecorrelation=False, doScaleVariance=False, scaleFactor=scaleFactor)

    def test_exposure_properties(self):
        """Check that all necessary exposure metadata is included
        with the Score image.
        """
        noiseLevel = 1.
        xSize = 400
        ySize = 400
        science, sources = makeTestImage(psfSize=3.0, noiseLevel=noiseLevel, noiseSeed=6,
                                         xSize=xSize, ySize=ySize)
        psf = science.psf
        psfAvgPos = psf.getAveragePosition()
        psfSize = getPsfFwhm(science.psf)
        psfImg = psf.computeKernelImage(psfAvgPos)
        template, _ = makeTestImage(psfSize=2.0, noiseLevel=noiseLevel, noiseSeed=7,
                                    templateBorderSize=20, doApplyCalibration=True,
                                    xSize=xSize, ySize=ySize)

        def _run_and_check_images(doDecorrelation):
            """Check that the metadata is correct with or without decorrelation.
            """
            task = self._setup_subtraction(doDecorrelation=doDecorrelation)
            output = task.run(template.clone(), science.clone(), sources)
            psfOut = output.scoreExposure.psf
            psfAvgPos = psfOut.getAveragePosition()
            if doDecorrelation:
                # Decorrelation requires recalculating the PSF,
                #  so it will not be the same as the input
                psfOutSize = getPsfFwhm(science.psf)
                self.assertFloatsAlmostEqual(psfSize, psfOutSize)
            else:
                psfOutImg = psfOut.computeKernelImage(psfAvgPos)
                self.assertImagesAlmostEqual(psfImg, psfOutImg)

            # check PSF, WCS, bbox, filterLabel, photoCalib
            self.assertWcsAlmostEqualOverBBox(science.wcs, output.scoreExposure.wcs, science.getBBox())
            self.assertEqual(science.filter, output.scoreExposure.filter)
            self.assertEqual(science.photoCalib, output.scoreExposure.photoCalib)
        _run_and_check_images(doDecorrelation=True)
        _run_and_check_images(doDecorrelation=False)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
