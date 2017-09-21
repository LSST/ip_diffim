from __future__ import absolute_import, division, print_function
#
# LSST Data Management System
# Copyright 2016 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.meas.algorithms as measAlg
import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.log

from .imageMapReduce import (ImageMapReduceConfig, ImageMapReduceTask,
                             ImageMapper)

__all__ = ("DecorrelateALKernelTask", "DecorrelateALKernelConfig",
           "DecorrelateALKernelMapper", "DecorrelateALKernelMapReduceConfig",
           "DecorrelateALKernelSpatialConfig", "DecorrelateALKernelSpatialTask")


class DecorrelateALKernelConfig(pexConfig.Config):
    """!
    \anchor DecorrelateALKernelConfig_

    \brief Configuration parameters for the DecorrelateALKernelTask
    """

    ignoreMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="""Mask planes to ignore for sigma-clipped statistics""",
        default=("INTRP", "EDGE", "DETECTED", "SAT", "CR", "BAD", "NO_DATA", "DETECTED_NEGATIVE")
    )

## \addtogroup LSST_task_documentation
## \{
## \page DecorrelateALKernelTask
## \ref DecorrelateALKernelTask_ "DecorrelateALKernelTask"
##      Decorrelate the effect of convolution by Alard-Lupton matching kernel in image difference
## \}


class DecorrelateALKernelTask(pipeBase.Task):
    """!
    \anchor DecorrelateALKernelTask_

    \brief Decorrelate the effect of convolution by Alard-Lupton matching kernel in image difference

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Contents      Contents

      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Purpose
      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Config
      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Run
      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Debug
      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Example

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Purpose	Description

    Pipe-task that removes the neighboring-pixel covariance in an
    image difference that are added when the template image is
    convolved with the Alard-Lupton PSF matching kernel.

    The image differencing pipeline task \link
    ip.diffim.psfMatch.PsfMatchTask PSFMatchTask\endlink and \link
    ip.diffim.psfMatch.PsfMatchConfigAL PSFMatchConfigAL\endlink uses
    the Alard and Lupton (1998) method for matching the PSFs of the
    template and science exposures prior to subtraction. The
    Alard-Lupton method identifies a matching kernel, which is then
    (typically) convolved with the template image to perform PSF
    matching. This convolution has the effect of adding covariance
    between neighboring pixels in the template image, which is then
    added to the image difference by subtraction.

    The pixel covariance may be corrected by whitening the noise of
    the image difference. This task performs such a decorrelation by
    computing a decorrelation kernel (based upon the A&L matching
    kernel and variances in the template and science images) and
    convolving the image difference with it. This process is described
    in detail in [DMTN-021](http://dmtn-021.lsst.io).

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Initialize       Task initialization

    \copydoc \_\_init\_\_

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Run       Invoking the Task

    \copydoc run

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Config       Configuration parameters

    See \ref DecorrelateALKernelConfig

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Debug		Debug variables

    This task has no debug variables

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelTask_Example	Example of using DecorrelateALKernelTask

    This task has no standalone example, however it is applied as a
    subtask of pipe.tasks.imageDifference.ImageDifferenceTask.
    """
    ConfigClass = DecorrelateALKernelConfig
    _DefaultName = "ip_diffim_decorrelateALKernel"

    def __init__(self, *args, **kwargs):
        """! Create the image decorrelation Task
        @param *args arguments to be passed to lsst.pipe.base.task.Task.__init__
        @param **kwargs keyword arguments to be passed to lsst.pipe.base.task.Task.__init__
        """
        pipeBase.Task.__init__(self, *args, **kwargs)

        self.statsControl = afwMath.StatisticsControl()
        self.statsControl.setNumSigmaClip(3.)
        self.statsControl.setNumIter(3)
        self.statsControl.setAndMask(afwImage.Mask\
                                     .getPlaneBitMask(self.config.ignoreMaskPlanes))

    def computeVarianceMean(self, exposure):
        statObj = afwMath.makeStatistics(exposure.getMaskedImage().getVariance(),
                                         exposure.getMaskedImage().getMask(),
                                         afwMath.MEANCLIP, self.statsControl)
        var = statObj.getValue(afwMath.MEANCLIP)
        return var

    @pipeBase.timeMethod
    def run(self, exposure, templateExposure, subtractedExposure, psfMatchingKernel,
            preConvKernel=None, xcen=None, ycen=None, svar=None, tvar=None):
        """! Perform decorrelation of an image difference exposure.

        Decorrelates the diffim due to the convolution of the templateExposure with the
        A&L PSF matching kernel. Currently can accept a spatially varying matching kernel but in
        this case it simply uses a static kernel from the center of the exposure. The decorrelation
        is described in [DMTN-021, Equation 1](http://dmtn-021.lsst.io/#equation-1), where
        `exposure` is I_1; templateExposure is I_2; `subtractedExposure` is D(k);
        `psfMatchingKernel` is kappa; and svar and tvar are their respective
        variances (see below).

        @param[in] exposure the science afwImage.Exposure used for PSF matching
        @param[in] templateExposure the template afwImage.Exposure used for PSF matching
        @param[in] subtractedExposure the subtracted exposure produced by
        `ip_diffim.ImagePsfMatchTask.subtractExposures()`
        @param[in] psfMatchingKernel an (optionally spatially-varying) PSF matching kernel produced
        by `ip_diffim.ImagePsfMatchTask.subtractExposures()`
        @param[in] preConvKernel if not None, then the `exposure` was pre-convolved with this kernel
        @param[in] xcen X-pixel coordinate to use for computing constant matching kernel to use
        If `None` (default), then use the center of the image.
        @param[in] ycen Y-pixel coordinate to use for computing constant matching kernel to use
        If `None` (default), then use the center of the image.
        @param[in] svar image variance for science image
        If `None` (default) then compute the variance over the entire input science image.
        @param[in] tvar image variance for template image
        If `None` (default) then compute the variance over the entire input template image.

        @return a `pipeBase.Struct` containing:
            * `correctedExposure`: the decorrelated diffim
            * `correctionKernel`: the decorrelation correction kernel (which may be ignored)

        @note The `subtractedExposure` is NOT updated
        @note The returned `correctedExposure` has an updated PSF as well.
        @note Here we currently convert a spatially-varying matching kernel into a constant kernel,
        just by computing it at the center of the image (tickets DM-6243, DM-6244).
        @note We are also using a constant accross-the-image measure of sigma (sqrt(variance)) to compute
        the decorrelation kernel.
        @note Still TBD (ticket DM-6580): understand whether the convolution is correctly modifying
        the variance plane of the new subtractedExposure.
        """
        spatialKernel = psfMatchingKernel
        kimg = afwImage.ImageD(spatialKernel.getDimensions())
        bbox = subtractedExposure.getBBox()
        if xcen is None:
            xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
        if ycen is None:
            ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
        self.log.info("Using matching kernel computed at (%d, %d)", xcen, ycen)
        spatialKernel.computeImage(kimg, True, xcen, ycen)

        if svar is None:
            svar = self.computeVarianceMean(exposure)
        if tvar is None:
            tvar = self.computeVarianceMean(templateExposure)
        self.log.info("Variance (science, template): (%f, %f)", svar, tvar)

        # Should not happen unless entire image has been masked, which could happen
        # if this is a small subimage of the main exposure. In this case, just return a full NaN
        # exposure
        if np.isnan(svar) or np.isnan(tvar):
            # Double check that one of the exposures is all NaNs
            if (np.all(np.isnan(exposure.getMaskedImage().getImage().getArray())) or
                    np.all(np.isnan(templateExposure.getMaskedImage().getImage().getArray()))):
                self.log.warn('Template or science image is entirely NaNs: skipping decorrelation.')
                outExposure = subtractedExposure.clone()
                return pipeBase.Struct(correctedExposure=outExposure, correctionKernel=None)

        var = self.computeVarianceMean(subtractedExposure)
        self.log.info("Variance (uncorrected diffim): %f", var)

        pck = None
        if preConvKernel is not None:
            self.log.info('Using a pre-convolution kernel as part of decorrelation.')
            kimg2 = afwImage.ImageD(preConvKernel.getDimensions())
            preConvKernel.computeImage(kimg2, False)
            pck = kimg2.getArray()
        corrKernel = DecorrelateALKernelTask._computeDecorrelationKernel(kimg.getArray(), svar, tvar,
                                                                         pck)
        correctedExposure, corrKern = DecorrelateALKernelTask._doConvolve(subtractedExposure, corrKernel)

        # Compute the subtracted exposure's updated psf
        psf = subtractedExposure.getPsf().computeKernelImage(afwGeom.Point2D(xcen, ycen)).getArray()
        psfc = DecorrelateALKernelTask.computeCorrectedDiffimPsf(corrKernel, psf, svar=svar, tvar=tvar)
        psfcI = afwImage.ImageD(psfc.shape[0], psfc.shape[1])
        psfcI.getArray()[:, :] = psfc
        psfcK = afwMath.FixedKernel(psfcI)
        psfNew = measAlg.KernelPsf(psfcK)
        correctedExposure.setPsf(psfNew)

        var = self.computeVarianceMean(correctedExposure)
        self.log.info("Variance (corrected diffim): %f", var)

        return pipeBase.Struct(correctedExposure=correctedExposure, correctionKernel=corrKern)

    @staticmethod
    def _computeDecorrelationKernel(kappa, svar=0.04, tvar=0.04, preConvKernel=None):
        """! Compute the Lupton decorrelation post-conv. kernel for decorrelating an
        image difference, based on the PSF-matching kernel.
        @param kappa  A matching kernel 2-d numpy.array derived from Alard & Lupton PSF matching
        @param svar   Average variance of science image used for PSF matching
        @param tvar   Average variance of template image used for PSF matching
        @param preConvKernel If not None, then pre-filtering was applied
            to science exposure, and this is the pre-convolution kernel.
        @return a 2-d numpy.array containing the correction kernel

        @note As currently implemented, kappa is a static (single, non-spatially-varying) kernel.
        """
        # Psf should not be <= 0, and messes up denominator; set the minimum value to MIN_KERNEL
        MIN_KERNEL = 1.0e-4

        kappa = DecorrelateALKernelTask._fixOddKernel(kappa)
        if preConvKernel is not None:
            mk = DecorrelateALKernelTask._fixOddKernel(preConvKernel)
            # Need to make them the same size
            if kappa.shape[0] < mk.shape[0]:
                diff = (mk.shape[0] - kappa.shape[0]) // 2
                kappa = np.pad(kappa, (diff, diff), mode='constant')
            elif kappa.shape[0] > mk.shape[0]:
                diff = (kappa.shape[0] - mk.shape[0]) // 2
                mk = np.pad(mk, (diff, diff), mode='constant')

        kft = np.fft.fft2(kappa)
        kft2 = np.conj(kft) * kft
        kft2[np.abs(kft2) < MIN_KERNEL] = MIN_KERNEL
        denom = svar + tvar * kft2
        if preConvKernel is not None:
            mk = np.fft.fft2(mk)
            mk2 = np.conj(mk) * mk
            mk2[np.abs(mk2) < MIN_KERNEL] = MIN_KERNEL
            denom = svar * mk2 + tvar * kft2
        denom[np.abs(denom) < MIN_KERNEL] = MIN_KERNEL
        kft = np.sqrt((svar + tvar) / denom)
        pck = np.fft.ifft2(kft)
        pck = np.fft.ifftshift(pck.real)
        fkernel = DecorrelateALKernelTask._fixEvenKernel(pck)
        if preConvKernel is not None:
            # This is not pretty but seems to be necessary as the preConvKernel term seems to lead
            # to a kernel that amplifies the noise way too much.
            fkernel[fkernel > -np.min(fkernel)] = -np.min(fkernel)

        # I think we may need to "reverse" the PSF, as in the ZOGY (and Kaiser) papers...
        # This is the same as taking the complex conjugate in Fourier space before FFT-ing back to real space.
        if False:  # TBD: figure this out. For now, we are turning it off.
            fkernel = fkernel[::-1, :]

        return fkernel

    @staticmethod
    def computeCorrectedDiffimPsf(kappa, psf, svar=0.04, tvar=0.04):
        """! Compute the (decorrelated) difference image's new PSF.
        new_psf = psf(k) * sqrt((svar + tvar) / (svar + tvar * kappa_ft(k)**2))

        @param kappa  A matching kernel array derived from Alard & Lupton PSF matching
        @param psf    The uncorrected psf array of the science image (and also of the diffim)
        @param svar   Average variance of science image used for PSF matching
        @param tvar   Average variance of template image used for PSF matching
        @return a 2-d numpy.array containing the new PSF
        """
        def post_conv_psf_ft2(psf, kernel, svar, tvar):
            # Pad psf or kernel symmetrically to make them the same size!
            # Note this assumes they are both square (width == height)
            if psf.shape[0] < kernel.shape[0]:
                diff = (kernel.shape[0] - psf.shape[0]) // 2
                psf = np.pad(psf, (diff, diff), mode='constant')
            elif psf.shape[0] > kernel.shape[0]:
                diff = (psf.shape[0] - kernel.shape[0]) // 2
                kernel = np.pad(kernel, (diff, diff), mode='constant')
            psf_ft = np.fft.fft2(psf)
            kft = np.fft.fft2(kernel)
            out = psf_ft * np.sqrt((svar + tvar) / (svar + tvar * kft**2))
            return out

        def post_conv_psf(psf, kernel, svar, tvar):
            kft = post_conv_psf_ft2(psf, kernel, svar, tvar)
            out = np.fft.ifft2(kft)
            return out

        pcf = post_conv_psf(psf=psf, kernel=kappa, svar=svar, tvar=tvar)
        pcf = pcf.real / pcf.real.sum()
        return pcf

    @staticmethod
    def _fixOddKernel(kernel):
        """! Take a kernel with odd dimensions and make them even for FFT

        @param kernel a numpy.array
        @return a fixed kernel numpy.array. Returns a copy if the dimensions needed to change;
        otherwise just return the input kernel.
        """
        # Note this works best for the FFT if we left-pad
        out = kernel
        changed = False
        if (out.shape[0] % 2) == 1:
            out = np.pad(out, ((1, 0), (0, 0)), mode='constant')
            changed = True
        if (out.shape[1] % 2) == 1:
            out = np.pad(out, ((0, 0), (1, 0)), mode='constant')
            changed = True
        if changed:
            out *= (np.mean(kernel) / np.mean(out))  # need to re-scale to same mean for FFT
        return out

    @staticmethod
    def _fixEvenKernel(kernel):
        """! Take a kernel with even dimensions and make them odd, centered correctly.
        @param kernel a numpy.array
        @return a fixed kernel numpy.array
        """
        # Make sure the peak (close to a delta-function) is in the center!
        maxloc = np.unravel_index(np.argmax(kernel), kernel.shape)
        out = np.roll(kernel, kernel.shape[0]//2 - maxloc[0], axis=0)
        out = np.roll(out, out.shape[1]//2 - maxloc[1], axis=1)
        # Make sure it is odd-dimensioned by trimming it.
        if (out.shape[0] % 2) == 0:
            maxloc = np.unravel_index(np.argmax(out), out.shape)
            if out.shape[0] - maxloc[0] > maxloc[0]:
                out = out[:-1, :]
            else:
                out = out[1:, :]
            if out.shape[1] - maxloc[1] > maxloc[1]:
                out = out[:, :-1]
            else:
                out = out[:, 1:]
        return out

    @staticmethod
    def _doConvolve(exposure, kernel):
        """! Convolve an Exposure with a decorrelation convolution kernel.
        @param exposure Input afw.image.Exposure to be convolved.
        @param kernel Input 2-d numpy.array to convolve the image with
        @return a new Exposure with the convolved pixels and the (possibly
        re-centered) kernel.

        @note We re-center the kernel if necessary and return the possibly re-centered kernel
        """
        kernelImg = afwImage.ImageD(kernel.shape[0], kernel.shape[1])
        kernelImg.getArray()[:, :] = kernel
        kern = afwMath.FixedKernel(kernelImg)
        maxloc = np.unravel_index(np.argmax(kernel), kernel.shape)
        kern.setCtrX(maxloc[0])
        kern.setCtrY(maxloc[1])
        outExp = exposure.clone()  # Do this to keep WCS, PSF, masks, etc.
        convCntrl = afwMath.ConvolutionControl(False, True, 0)
        afwMath.convolve(outExp.getMaskedImage(), exposure.getMaskedImage(), kern, convCntrl)

        return outExp, kern


class DecorrelateALKernelMapper(DecorrelateALKernelTask, ImageMapper):
    """Task to be used as an ImageMapper for performing
    A&L decorrelation on subimages on a grid across a A&L difference image.

    This task subclasses DecorrelateALKernelTask in order to implement
    all of that task's configuration parameters, as well as its `run` method.
    """
    ConfigClass = DecorrelateALKernelConfig
    _DefaultName = 'ip_diffim_decorrelateALKernelMapper'

    def __init__(self, *args, **kwargs):
        DecorrelateALKernelTask.__init__(self, *args, **kwargs)

    def run(self, subExposure, expandedSubExposure, fullBBox,
            template, science, alTaskResult=None, psfMatchingKernel=None,
            preConvKernel=None, **kwargs):
        """Perform decorrelation operation on `subExposure`, using
        `expandedSubExposure` to allow for invalid edge pixels arising from
        convolutions.

        This method performs A&L decorrelation on `subExposure` using
        local measures for image variances and PSF. `subExposure` is a
        sub-exposure of the non-decorrelated A&L diffim. It also
        requires the corresponding sub-exposures of the template
        (`template`) and science (`science`) exposures.

        Parameters
        ----------
        subExposure : lsst.afw.image.Exposure
            the sub-exposure of the diffim
        expandedSubExposure : lsst.afw.image.Exposure
            the expanded sub-exposure upon which to operate
        fullBBox : afwGeom.BoundingBox
            the bounding box of the original exposure
        template : afw.Exposure
            the corresponding sub-exposure of the template exposure
        science : afw.Exposure
            the corresponding sub-exposure of the science exposure
        alTaskResult : pipeBase.Struct
            the result of A&L image differencing on `science` and
            `template`, importantly containing the resulting
            `psfMatchingKernel`. Can be `None`, only if
            `psfMatchingKernel` is not `None`.
        psfMatchingKernel : Alternative parameter for passing the
            A&L `psfMatchingKernel` directly.
        preConvKernel : If not None, then pre-filtering was applied
            to science exposure, and this is the pre-convolution
            kernel.
        kwargs :
            additional keyword arguments propagated from
            `ImageMapReduceTask.run`.

        Returns
        -------
        A `pipeBase.Struct` containing:
        * `subExposure` : the result of the `subExposure` processing.
        * `decorrelationKernel` : the decorrelation kernel, currently
          not used.

        Notes
        -----
        This `run` method accepts parameters identical to those of
        `ImageMapper.run`, since it is called from the
        `ImageMapperTask`. See that class for more information.
        """
        templateExposure = template  # input template
        scienceExposure = science  # input science image
        if alTaskResult is None and psfMatchingKernel is None:
            raise RuntimeError('Both alTaskResult and psfMatchingKernel cannot be None')
        psfMatchingKernel = alTaskResult.psfMatchingKernel if alTaskResult is not None else psfMatchingKernel

        # subExp and expandedSubExp are subimages of the (un-decorrelated) diffim!
        # So here we compute corresponding subimages of templateExposure and scienceExposure
        subExp2 = scienceExposure.Factory(scienceExposure, expandedSubExposure.getBBox())
        subExp1 = templateExposure.Factory(templateExposure, expandedSubExposure.getBBox())

        # Prevent too much log INFO verbosity from DecorrelateALKernelTask.run
        logLevel = self.log.getLevel()
        self.log.setLevel(lsst.log.WARN)
        res = DecorrelateALKernelTask.run(self, subExp2, subExp1, expandedSubExposure,
                                          psfMatchingKernel, preConvKernel)
        self.log.setLevel(logLevel)  # reset the log level

        diffim = res.correctedExposure.Factory(res.correctedExposure, subExposure.getBBox())
        out = pipeBase.Struct(subExposure=diffim, decorrelationKernel=res.correctionKernel)
        return out


class DecorrelateALKernelMapReduceConfig(ImageMapReduceConfig):
    """Configuration parameters for the ImageMapReduceTask to direct it to use
       DecorrelateALKernelMapper as its mapper for A&L decorrelation.
    """
    mapper = pexConfig.ConfigurableField(
        doc='A&L decorrelation task to run on each sub-image',
        target=DecorrelateALKernelMapper
    )


## \addtogroup LSST_task_documentation
## \{
## \page DecorrelateALKernelSpatialTask
## \ref DecorrelateALKernelSpatialTask_ "DecorrelateALKernelSpatialTask"
##      Decorrelate the effect of convolution by Alard-Lupton matching kernel in image difference,
##      allowing for spatial variations in PSF and noise
## \}


class DecorrelateALKernelSpatialConfig(pexConfig.Config):
    """Configuration parameters for the DecorrelateALKernelSpatialTask.
    """
    decorrelateConfig = pexConfig.ConfigField(
        dtype=DecorrelateALKernelConfig,
        doc='DecorrelateALKernel config to use when running on complete exposure (non spatially-varying)',
    )

    decorrelateMapReduceConfig = pexConfig.ConfigField(
        dtype=DecorrelateALKernelMapReduceConfig,
        doc='DecorrelateALKernelMapReduce config to use when running on each sub-image (spatially-varying)',
    )

    ignoreMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="""Mask planes to ignore for sigma-clipped statistics""",
        default=("INTRP", "EDGE", "DETECTED", "SAT", "CR", "BAD", "NO_DATA", "DETECTED_NEGATIVE")
    )

    def setDefaults(self):
        self.decorrelateMapReduceConfig.gridStepX = self.decorrelateMapReduceConfig.gridStepY = 40
        self.decorrelateMapReduceConfig.cellSizeX = self.decorrelateMapReduceConfig.cellSizeY = 41
        self.decorrelateMapReduceConfig.borderSizeX = self.decorrelateMapReduceConfig.borderSizeY = 8
        self.decorrelateMapReduceConfig.reducer.reduceOperation = 'average'


class DecorrelateALKernelSpatialTask(pipeBase.Task):
    """!
    \anchor DecorrelateALKernelSpatialTask_

    \brief Decorrelate the effect of convolution by Alard-Lupton matching kernel in image difference

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Contents       Contents

      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Purpose
      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Config
      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Run
      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Debug
      - \ref ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Example

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Purpose	Description

    Pipe-task that removes the neighboring-pixel covariance in an
    image difference that are added when the template image is
    convolved with the Alard-Lupton PSF matching kernel.

    This task is a simple wrapper around \ref DecorrelateALKernelTask,
    which takes a `spatiallyVarying` parameter in its `run` method. If
    it is `False`, then it simply calls the `run` method of \ref
    DecorrelateALKernelTask. If it is True, then it uses the \ref
    ImageMapReduceTask framework to break the exposures into
    subExposures on a grid, and performs the `run` method of \ref
    DecorrelateALKernelTask on each subExposure. This enables it to
    account for spatially-varying PSFs and noise in the exposures when
    performing the decorrelation.

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Initialize       Task initialization

    \copydoc \_\_init\_\_

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Run       Invoking the Task

    \copydoc run

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Config       Configuration parameters

    See \ref DecorrelateALKernelSpatialConfig

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Debug		Debug variables

    This task has no debug variables

    \section ip_diffim_imageDecorrelation_DecorrelateALKernelSpatialTask_Example	Example of using DecorrelateALKernelSpatialTask

    This task has no standalone example, however it is applied as a
    subtask of pipe.tasks.imageDifference.ImageDifferenceTask.
    There is also an example of its use in `tests/testImageDecorrelation.py`.
    """
    ConfigClass = DecorrelateALKernelSpatialConfig
    _DefaultName = "ip_diffim_decorrelateALKernelSpatial"

    def __init__(self, *args, **kwargs):
        """Create the image decorrelation Task

        Parameters
        ----------
        args :
            arguments to be passed to
            `lsst.pipe.base.task.Task.__init__`
        kwargs :
            additional keyword arguments to be passed to
            `lsst.pipe.base.task.Task.__init__`
        """
        pipeBase.Task.__init__(self, *args, **kwargs)

        self.statsControl = afwMath.StatisticsControl()
        self.statsControl.setNumSigmaClip(3.)
        self.statsControl.setNumIter(3)
        self.statsControl.setAndMask(afwImage.Mask\
                                     .getPlaneBitMask(self.config.ignoreMaskPlanes))

    def computeVarianceMean(self, exposure):
        """Compute the mean of the variance plane of `exposure`.
        """
        statObj = afwMath.makeStatistics(exposure.getMaskedImage().getVariance(),
                                         exposure.getMaskedImage().getMask(),
                                         afwMath.MEANCLIP, self.statsControl)
        var = statObj.getValue(afwMath.MEANCLIP)
        return var

    def run(self, scienceExposure, templateExposure, subtractedExposure, psfMatchingKernel,
            spatiallyVarying=True, preConvKernel=None):
        """! Perform decorrelation of an image difference exposure.

        Decorrelates the diffim due to the convolution of the
        templateExposure with the A&L psfMatchingKernel. If
        `spatiallyVarying` is True, it utilizes the spatially varying
        matching kernel via the `imageMapReduce` framework to perform
        spatially-varying decorrelation on a grid of subExposures.

        Parameters
        ----------
        scienceExposure : lsst.afw.image.Exposure
           the science Exposure used for PSF matching
        templateExposure : lsst.afw.image.Exposure
           the template Exposure used for PSF matching
        subtractedExposure : lsst.afw.image.Exposure
           the subtracted Exposure produced by `ip_diffim.ImagePsfMatchTask.subtractExposures()`
        psfMatchingKernel :
           an (optionally spatially-varying) PSF matching kernel produced
           by `ip_diffim.ImagePsfMatchTask.subtractExposures()`
        spatiallyVarying : bool
           if True, perform the spatially-varying operation
        preConvKernel : lsst.meas.algorithms.Psf
           if not none, the scienceExposure has been pre-filtered with this kernel. (Currently
           this option is experimental.)

        Returns
        -------
        a `pipeBase.Struct` containing:
            * `correctedExposure`: the decorrelated diffim
        """
        self.log.info('Running A&L decorrelation: spatiallyVarying=%r' % spatiallyVarying)

        svar = self.computeVarianceMean(scienceExposure)
        tvar = self.computeVarianceMean(templateExposure)
        if np.isnan(svar) or np.isnan(tvar):  # Should not happen unless entire image has been masked.
            # Double check that one of the exposures is all NaNs
            if (np.all(np.isnan(scienceExposure.getMaskedImage().getImage().getArray())) or
                    np.all(np.isnan(templateExposure.getMaskedImage().getImage().getArray()))):
                self.log.warn('Template or science image is entirely NaNs: skipping decorrelation.')
                if np.isnan(svar):
                    svar = 1e-9
                if np.isnan(tvar):
                    tvar = 1e-9

        var = self.computeVarianceMean(subtractedExposure)

        if spatiallyVarying:
            self.log.info("Variance (science, template): (%f, %f)", svar, tvar)
            self.log.info("Variance (uncorrected diffim): %f", var)
            config = self.config.decorrelateMapReduceConfig
            task = ImageMapReduceTask(config=config)
            results = task.run(subtractedExposure, science=scienceExposure,
                               template=templateExposure, psfMatchingKernel=psfMatchingKernel,
                               preConvKernel=preConvKernel, forceEvenSized=True)
            results.correctedExposure = results.exposure

            # Make sure masks of input image are propagated to diffim
            def gm(exp):
                return exp.getMaskedImage().getMask()
            gm(results.correctedExposure)[:, :] = gm(subtractedExposure)

            var = self.computeVarianceMean(results.correctedExposure)
            self.log.info("Variance (corrected diffim): %f", var)

        else:
            config = self.config.decorrelateConfig
            task = DecorrelateALKernelTask(config=config)
            results = task.run(scienceExposure, templateExposure,
                               subtractedExposure, psfMatchingKernel, preConvKernel=preConvKernel)

        return results
