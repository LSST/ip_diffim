#
# LSST Data Management System
# Copyright 2008-2016 LSST Corporation.
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#

__all__ = ["makeKernelBasisList", "generateAlardLuptonBasisList"]

from . import diffimLib
import numpy as np

from lsst.utils.logging import getLogger

sigma2fwhm = 2. * np.sqrt(2. * np.log(2.))


def makeKernelBasisList(config, targetFwhmPix=None, referenceFwhmPix=None,
                        basisDegGauss=None, basisSigmaGauss=None, metadata=None):
    """Generate the delta function or Alard-Lupton kernel bases depending on the Config.
    Wrapper to call either `lsst.ip.diffim.makeDeltaFunctionBasisList` or
    `lsst.ip.diffim.generateAlardLuptonBasisList`.

    Parameters
    ----------
    config : `lsst.ip.diffim.PsfMatchConfigAL`
        Configuration object.
    targetFwhmPix : `float`, optional
        Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
        Not used for delta function basis sets.
    referenceFwhmPix : `float`, optional
        Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
        Not used for delta function basis sets.
    basisDegGauss : `list` of `int`, optional
        Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
        Not used for delta function basis sets.
    basisSigmaGauss : `list` of `int`, optional
        Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
        Not used for delta function basis sets.
    metadata : `lsst.daf.base.PropertySet`, optional
        Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
        Not used for delta function basis sets.

    Returns
    -------
    basisList: `list` of `lsst.afw.math.kernel.FixedKernel`
        List of basis kernels.

    Notes
    -----
    See `lsst.ip.diffim.generateAlardLuptonBasisList` and
    `lsst.ip.diffim.makeDeltaFunctionBasisList` for more information.

    Raises
    ------
    ValueError
        If ``config.kernelBasisSet`` has an invalid value (not "alard-lupton" or "delta-function").
    """
    if config.kernelBasisSet == "alard-lupton":
        basis = generateAlardLuptonBasisList(config, targetFwhmPix=targetFwhmPix,
                                            referenceFwhmPix=referenceFwhmPix,
                                            basisDegGauss=basisDegGauss,
                                            basisSigmaGauss=basisSigmaGauss,
                                            metadata=metadata)
        return basis, len(basis)
    elif config.kernelBasisSet == "delta-function":
        kernelSize = config.kernelSize
        return diffimLib.makeDeltaFunctionBasisList(kernelSize, kernelSize), config.kernelSize
    elif config.kernelBasisSet == "combined":
        basisAL = generateAlardLuptonBasisList(config.psfMatchAL, targetFwhmPix=targetFwhmPix,
                                               referenceFwhmPix=referenceFwhmPix,
                                               basisDegGauss=basisDegGauss,
                                               basisSigmaGauss=basisSigmaGauss,
                                               metadata=metadata)
        kernelSize = config.psfMatchDF.kernelSize
        basisDF = diffimLib.makeDeltaFunctionBasisList(kernelSize, kernelSize)
        return basisAL+basisDF, (len(basisAL), len(basisDF))
    else:
        raise ValueError("Cannot generate %s basis set" % (config.kernelBasisSet))


def generateAlardLuptonBasisList(config, targetFwhmPix=None, referenceFwhmPix=None,
                                 basisDegGauss=None, basisSigmaGauss=None, metadata=None):
    """Generate an Alard-Lupton kernel basis list based upon the Config and
    the input FWHM of the science and template images.

    Parameters
    ----------
    config : `lsst.ip.diffim.PsfMatchConfigAL`
        Configuration object for the Alard-Lupton algorithm.
    targetFwhmPix : `float`, optional
        Fwhm width (pixel) of the template exposure characteristic psf.
        This is the _target_ that will be matched to the science exposure.
    referenceFwhmPix : `float`, optional
        Fwhm width (pixel) of the science exposure characteristic psf.
    basisDegGauss : `list` of `int`, optional
        Polynomial degree of each Gaussian (sigma) basis. If None, defaults to `config.alardDegGauss`.
    basisSigmaGauss : `list` of `int`, optional
        Sigmas of each Gaussian basis. If None, defaults to `config.alardSigGauss`.
    metadata : `lsst.daf.base.PropertySet`, optional
        If specified, object to collect metadata fields about the kernel basis list.

    Returns
    -------
    basisList : `list` of `lsst.afw.math.kernel.FixedKernel`
        List of basis kernels. For each degree value ``n`` in ``config.basisDegGauss`` (n+2)(n+1)/2 kernels
        are generated and appended to the list in the order of the polynomial parameter number.
        See `lsst.afw.math.polynomialFunction2D` documentation for more details.

    Raises
    ------
    RuntimeError
        - if ``config.kernelBasisSet`` is not equal to "alard-lupton"
    ValueError
        - if ``config.kernelSize`` is even
        - if the number of Gaussians and the number of given
          sigma values are not equal or
        - if the number of Gaussians and the number of given
          polynomial degree values are not equal

    Notes
    -----
    The polynomial functions (``f``) are always evaluated in the -1.0, +1.0 range in both x, y directions,
    edge to edge, with ``f(0,0)`` evaluated at the kernel center pixel, ``f(-1.0,-1.0)`` at the kernel
    ``(0,0)`` pixel. They are not scaled by the sigmas of the Gaussians.

    Base Gaussian widths (sigmas in pixels) of the kernels are determined as:
        - If not all fwhm parameters are provided or ``config.scaleByFwhm==False``
          then ``basisSigmaGauss`` is used. If ``basisSigmaGauss`` is not
          provided, then ``config.alardSigGauss`` is used. In both cases, the
          length of sigmas must be equal to ``config.alardNGauss``.
        - If ``targetFwhmPix<referenceFwhmPix`` (normal convolution):
          First sigma ``Sig_K`` is determined to satisfy: ``Sig_reference**2 = Sig_target**2 + Sig_K**2``.
          If it's larger than ``config.alardMinSig * config.alardGaussBeta``, make it the
          second kernel. Else make it the smallest kernel, unless only 1 kernel is asked for.
        - If ``referenceFwhmPix < targetFwhmPix`` (deconvolution):
          Define the progression of Gaussians using a
          method to derive a deconvolution sum-of-Gaussians from it's
          convolution counterpart. [1]_ Only use 3 since the algorithm
          assumes 3 components.

    **Metadata fields**

    ALBasisNGauss : `int`
        The number of base Gaussians in the AL basis functions.
    ALBasisDegGauss : `list` of `int`
        Polynomial order of spatial modification of the base Gaussian functions.
    ALBasisSigGauss : `list` of `float`
        Sigmas in pixels of the base Gaussians.
    ALKernelSize : `int`
        Kernel stamp size is (ALKernelSize pix, ALKernelSize pix).
    ALBasisMode : `str`, either of ``config``, ``convolution``, ``deconvolution``
        Indicates whether the config file values, the convolution or deconvolution algorithm
        was used to determine the base Gaussian sigmas and the kernel stamp size.

    References
    ----------
    .. [1] Ulmer, W.: Inverse problem of linear combinations of Gaussian convolution kernels
       (deconvolution) and some applications to proton/photon dosimetry and image
       processing. http://iopscience.iop.org/0266-5611/26/8/085002  Equation 40
    """

    if config.kernelBasisSet != "alard-lupton":
        raise RuntimeError("Cannot generate %s basis within generateAlardLuptonBasisList" %
                           config.kernelBasisSet)

    kernelSize = config.kernelSize
    fwhmScaling = config.kernelSizeFwhmScaling
    basisNGauss = config.alardNGauss
    basisGaussBeta = config.alardGaussBeta
    basisMinSigma = config.alardMinSig
    if basisDegGauss is None:
        basisDegGauss = config.alardDegGauss
    if basisSigmaGauss is None:
        basisSigmaGauss = config.alardSigGauss

    if len(basisDegGauss) != basisNGauss:
        raise ValueError("len(basisDegGauss) != basisNGauss : %d vs %d" % (len(basisDegGauss), basisNGauss))
    if len(basisSigmaGauss) != basisNGauss:
        raise ValueError("len(basisSigmaGauss) != basisNGauss : %d vs %d" %
                         (len(basisSigmaGauss), basisNGauss))
    if (kernelSize % 2) != 1:
        raise ValueError("Only odd-sized Alard-Lupton bases allowed")

    logger = getLogger("lsst.ip.diffim.generateAlardLuptonBasisList")
    if (targetFwhmPix is None) or (referenceFwhmPix is None) or (not config.scaleByFwhm):
        logger.info("PSF sigmas are not available or scaling by fwhm disabled, "
                    "falling back to config values")
        if metadata is not None:
            metadata.add("ALBasisNGauss", basisNGauss)
            metadata.add("ALBasisDegGauss", basisDegGauss)
            metadata.add("ALBasisSigGauss", basisSigmaGauss)
            metadata.add("ALKernelSize", kernelSize)
            metadata.add("ALBasisMode", "config")

        return diffimLib.makeAlardLuptonBasisList(kernelSize//2, basisNGauss, basisSigmaGauss, basisDegGauss)

    targetSigma = targetFwhmPix / sigma2fwhm
    referenceSigma = referenceFwhmPix / sigma2fwhm
    logger.debug("Generating matching bases for sigma %.2f pix -> %.2f pix", targetSigma, referenceSigma)

    # Modify the size of Alard Lupton kernels based upon the images FWHM
    #
    # Note the operation is : template.x.kernel = science
    #
    # Assuming the template and science image Psfs are Gaussians with
    # the Fwhm above, Fwhm_T **2 + Fwhm_K **2 = Fwhm_S **2
    #
    if targetSigma == referenceSigma:
        # Leave defaults as-is
        logger.debug("Target and reference psf fwhms are equal, falling back to config values")
        basisMode = "config"
    elif referenceSigma > targetSigma:
        # Normal convolution

        # First Gaussian has the sigma that comes from the convolution
        # of two Gaussians : Sig_S**2 = Sig_T**2 + Sig_K**2
        #
        # If it's larger than basisMinSigma * basisGaussBeta, make it the
        # second kernel.  Else make it the smallest kernel.  Unless
        # only 1 kernel is asked for.
        logger.debug("Reference psf fwhm is the greater, normal convolution mode")
        basisMode = "convolution"
        kernelSigma = np.sqrt(referenceSigma**2 - targetSigma**2)
        if kernelSigma < basisMinSigma:
            kernelSigma = basisMinSigma

        basisSigmaGauss = []
        if basisNGauss == 1:
            basisSigmaGauss.append(kernelSigma)
            nAppended = 1
        else:
            if (kernelSigma/basisGaussBeta) > basisMinSigma:
                basisSigmaGauss.append(kernelSigma/basisGaussBeta)
                basisSigmaGauss.append(kernelSigma)
                nAppended = 2
            else:
                basisSigmaGauss.append(kernelSigma)
                nAppended = 1

        # Any other Gaussians above basisNGauss=1 come from a scaling
        # relationship: Sig_i+1 / Sig_i = basisGaussBeta
        for i in range(nAppended, basisNGauss):
            basisSigmaGauss.append(basisSigmaGauss[-1]*basisGaussBeta)

        kernelSize = int(fwhmScaling * basisSigmaGauss[-1])
        kernelSize += 0 if kernelSize%2 else 1  # Make sure it's odd
        kernelSize = min(config.kernelSizeMax, max(kernelSize, config.kernelSizeMin))

    else:
        # Deconvolution; Define the progression of Gaussians using a
        # method to derive a deconvolution sum-of-Gaussians from it's
        # convolution counterpart.  Only use 3 since the algorithm
        # assumes 3 components.
        #
        # http://iopscience.iop.org/0266-5611/26/8/085002  Equation 40

        # Use specializations for deconvolution
        logger.debug("Target psf fwhm is the greater, deconvolution mode")
        basisMode = "deconvolution"
        basisNGauss = config.alardNGaussDeconv
        basisMinSigma = config.alardMinSigDeconv

        kernelSigma = np.sqrt(targetSigma**2 - referenceSigma**2)
        if kernelSigma < basisMinSigma:
            kernelSigma = basisMinSigma

        basisSigmaGauss = []
        if (kernelSigma/basisGaussBeta) > basisMinSigma:
            basisSigmaGauss.append(kernelSigma/basisGaussBeta)
            basisSigmaGauss.append(kernelSigma)
            nAppended = 2
        else:
            basisSigmaGauss.append(kernelSigma)
            nAppended = 1

        for i in range(nAppended, basisNGauss):
            basisSigmaGauss.append(basisSigmaGauss[-1]*basisGaussBeta)

        kernelSize = int(fwhmScaling * basisSigmaGauss[-1])
        kernelSize += 0 if kernelSize%2 else 1  # Make sure it's odd
        kernelSize = min(config.kernelSizeMax, max(kernelSize, config.kernelSizeMin))

        # Now build a deconvolution set from these sigmas
        sig0 = basisSigmaGauss[0]
        sig1 = basisSigmaGauss[1]
        sig2 = basisSigmaGauss[2]
        basisSigmaGauss = []
        for n in range(1, 3):
            for j in range(n):
                sigma2jn = (n - j)*sig1**2
                sigma2jn += j * sig2**2
                sigma2jn -= (n + 1)*sig0**2
                sigmajn = np.sqrt(sigma2jn)
                basisSigmaGauss.append(sigmajn)

        basisSigmaGauss.sort()
        basisNGauss = len(basisSigmaGauss)
        basisDegGauss = [config.alardDegGaussDeconv for x in basisSigmaGauss]

    if metadata is not None:
        metadata.add("ALBasisNGauss", basisNGauss)
        metadata.add("ALBasisDegGauss", basisDegGauss)
        metadata.add("ALBasisSigGauss", basisSigmaGauss)
        metadata.add("ALKernelSize", kernelSize)
        metadata.add("ALBasisMode", basisMode)

    logger.debug("basisSigmaGauss: %s basisDegGauss: %s",
                 ','.join(['{:.1f}'.format(v) for v in basisSigmaGauss]),
                 ','.join(['{:d}'.format(v) for v in basisDegGauss]))

    return diffimLib.makeAlardLuptonBasisList(kernelSize//2, basisNGauss, basisSigmaGauss, basisDegGauss)
