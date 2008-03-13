#!/usr/bin/env python
import numpy as num

import lsst.mwi.utils as mwiu
import lsst.mwi.exceptions as mwex
import lsst.mwi.policy
from lsst.mwi.logging import Log

import lsst.fw.Core.fwLib as fw
import lsst.detection.detectionLib as detection
import imageprocLib
from computePsfMatchingKernelForMaskedImage import *

__all__ = ['imageSubtract']

def getCollectionOfFootprintsForPsfMatching(imageToConvolve, imageToNotConvolve, policy):
    # hack until I can append to Vector2i or grow Footprint
    return imageprocLib.getCollectionOfFootprintsForPsfMatching(imageToConvolve, imageToNotConvolve, policy)
                                                             
    footprintDiffimNpixMin = policy.get('getCollectionOfFootprintsForPsfMatching.footprintDiffimNpixMin');
    footprintDiffimGrow = policy.get('getCollectionOfFootprintsForPsfMatching.footprintDiffimGrow');
    minimumCleanFootprints = policy.get('getCollectionOfFootprintsForPsfMatching.minimumCleanFootprints');
    footprintDetectionThresholdSigma = policy.get('getCollectionOfFootprintsForPsfMatching.footprintDetectionThresholdSigma');
    detectionThresholdScalingSigma = policy.get('getCollectionOfFootprintsForPsfMatching.detectionThresholdScalingSigma');
    minimumDetectionThresholdSigma = policy.get('getCollectionOfFootprintsForPsfMatching.minimumDetectionThresholdSigma');

    badMaskBit = imageToConvolve.getMask().getPlaneBitMask('BAD')
    badPixelMask = 1 << badMaskBit

    varImg = imageToConvolve.getVariance()
    noise = num.sqrt(fw.mean_channel_value(varImg))

    nCleanFootprints = 0
    while ( (nCleanFootprints < minimumCleanFootprints) and (footprintDetectionThresholdSigma >= minimumDetectionThresholdSigma) ):

        mwiu.Trace('lsst.imageproc.getCollectionOfFootprintsForPsfMatching', 3,
                   'thresholdSigma = %r; noise = %r; PixMin = %r' % (footprintDetectionThresholdSigma, noise, footprintDiffimNpixMin))
        detectionSet = detection.DetectionSetF(imageToConvolve,
                                               detection.Threshold(footprintDetectionThresholdSigma*noise, detection.Threshold.VALUE, True),
                                               'DIFP',
                                               footprintDiffimNpixMin)
        
        footprintListIn  = detectionSet.getFootprints()
        footprintListOut = detection.FootprintContainerT()
        
        nCleanFootprints = 0
        for footprintID, iFootprintPtr in enumerate(footprintListIn):
            footprintBBox = iFootprintPtr.getBBox()

            footprintGrow = footprintBBox.max()
            print footprintGrow, footprintGrow.__dict__
            footprintGrow[0] += footprintDiffimGrow
            footprintGrow[1] += footprintDiffimGrow
            footprintBBox.grow(footprintGrow)
            
            footprintGrow = footprintBBox.min()
            footprintGrow[0] -= footprintDiffimGrow
            footprintGrow[1] -= footprintDiffimGrow
            footprintBBox.grow(footprintGrow)

            try:
                imageToConvolveFootprintPtr = imageToConvolve.getSubImage(footprintBBox);
                imageToNotConvolveFootprintPtr = imageToNotConvolve.getSubImage(footprintBBox);
            except wex.LsstExceptionStack, e:
                continue

            if ( lsst.imageproc.maskOk(imageToConvolve.getMask(), badPixelMask) and
                 lsst.imageproc.maskOk(imageToNotConvolve.getMask(), badPixelMask) ):

                footprintGrow = detection.FootprintPtrT(detection.Footprint(footprintBBox))
                footprintListOut.push_back(fpGrow)
                
                nCleanFootprints += 1;
            
        mwiu.Trace('lsst.imageproc.getCollectionOfFootprintsForPsfMatching', 3,
                   'Found %d clean footprints above threshold %.3f' % (footprintListOut.size(), footprintDetectionThresholdSigma*noise))
        
        footprintDetectionThresholdSigma -= detectionThresholdScalingSigma


    return footprintListOut

            
        

def imageSubtract(imageToConvolve, imageToNotConvolve, policy,
    psfMatchBasisKernelSet=None, footprintList=None):
    """Subtract two masked images after psf-matching them.
    
    Computes Idiff = Inc + B - Ic.conv.Kpsf
    where:
    - Idiff is the difference masked image
    - Inc is the masked image to not convolve
    - Ic is the masked image to convolve
    - B is the background (an output)
    - Kpsf is the psf-matching kernel (an output)
    
    Eventually imageToConvolve will also be wcs-matched to imageToNotConvolve,
    but for now that step must already have been performed.
    
    Inputs:
    - imageToConvolve: an lsst.fw.MaskedImage(x)
    - imageToConvolve: an lsst.fw.MaskedImage(x)
    - policy: the policy; required elements are...?
    - psfMatchBasisKernelSet: a sequence of kernel basis vectors; if omitted then a delta function kernel is used.
    - footprintList: a squence of detection footprints to use for computing
        the psf-matching convolution kernel; a detection.FootprintContainerT.
        If omitted then the footprints are found automatically.
        
    Returns:
    - differenceMaskedImage: the difference masked image (Idiff in the equation above)
    - psfMatchKernelPtr: pointer to psf matching kernel (Kpsf in the equation above)
    - backgroundFunctionPtr: pointer to function representation of the background (B in the equation above)
    """
    ###########
    #
    # Get directives from policy
    #
    kernelCols = policy.get('kernelCols')
    kernelRows = policy.get('kernelRows')
    kernelSpatialOrder = policy.get('kernelSpatialOrder')
    backgroundSpatialOrder = policy.get('backgroundSpatialOrder')
    badPixelMask = imageToConvolve.getMask().getPlaneBitMask('BAD') \
        | imageToConvolve.getMask().getPlaneBitMask('EDGE')
    edgeMaskBit = imageToConvolve.getMask().getMaskPlane('EDGE')
    debugIO = policy.get('debugIO', False)
    switchConvolve = policy.get('switchConvolve', False)
    
    mwiu.Trace('lsst.imageproc.imageSubtract', 3,
        'kernelCols = %r; kernelRows = %r' % (kernelCols, kernelRows))
    mwiu.Trace('lsst.imageproc.imageSubtract', 3,
        'kernelSpatialOrder = %r; backgroundSpatialOrder = %r' % (kernelSpatialOrder, backgroundSpatialOrder))
    mwiu.Trace('lsst.imageproc.imageSubtract', 3,
        'edgeMaskBit = %r; badPixelMask = %r' % (edgeMaskBit, badPixelMask))

    ###########
    #
    # Generate objects from policy directives
    #
    
    # create basis vectors
    if psfMatchBasisKernelSet == None:
        psfMatchBasisKernelSet = imageprocLib.generateDeltaFunctionKernelSetD(kernelCols, kernelRows)
    
    # create function for kernel spatial variation
    kernelSpatialFunctionPtr = fw.Function2DPtr(fw.PolynomialFunction2D(kernelSpatialOrder))
    
    # and function for background
    backgroundFunctionPtr = fw.Function2DPtr(fw.PolynomialFunction2D(backgroundSpatialOrder))

    # get Log 
    diffImLog = Log(Log.getDefaultLog(), "imageproc.imageSubtract")

    if footprintList == None:
        footprintList = getCollectionOfFootprintsForPsfMatching(imageToConvolve, imageToNotConvolve, policy)
    else:
        mwiu.Trace('lsst.imageproc.imageSubtract', 3, 'User supplied %d footprints' % len(footprintList))
    mwiu.Trace('lsst.imageproc.imageSubtract', 4, "Computing psf-matching kernel")

    if switchConvolve:
        imageTmp = imageToConvolve
        imageToConvolve = imageToNotConvolve
        imageToNotConvolve = imageTmp

    psfMatchKernelPtr = computePsfMatchingKernelForMaskedImage(
        kernelSpatialFunctionPtr,
        backgroundFunctionPtr,
        imageToConvolve,
        imageToNotConvolve,
        psfMatchBasisKernelSet,
        footprintList,
        policy,
        )
    
    #
    # Create final difference image
    # MaskedImage cannot directly subtract two images, but it does support -= and +=
    # so compute diffIm = imageToNotConvolve - (background + convolvedImage) as follows:
    # diffIm  = convolvedImage (a newly created masked image)
    # diffIm += background (only the image component)
    # diffIm -= imageToNotConvolve
    # diffIm *= -1
    #
    if type(psfMatchKernelPtr.get()) == fw.LinearCombinationKernelD:
        mwiu.Trace('lsst.imageproc.imageSubtract', 4, "Psf-match using convolveLinear")
        differenceImage = fw.convolveLinear(imageToConvolve, psfMatchKernelPtr.get(), edgeMaskBit)
    else:
        mwiu.Trace('lsst.imageproc.imageSubtract', 4, "Psf-match using convolve")
        differenceImage = fw.convolve(imageToConvolve, psfMatchKernelPtr.get(), edgeMaskBit, False)

    mwiu.Trace('lsst.imageproc.imageSubtract', 4, "Add background")
    imageprocLib.addFunction(differenceImage.getImage().get(), backgroundFunctionPtr.get())
    mwiu.Trace('lsst.imageproc.imageSubtract', 4, "Subtract imageToNotConvolve")
    differenceImage      -= imageToNotConvolve
    differenceImage      *= -1.0

    # Find quality metrics
    nGoodPixels, meanOfResiduals, varianceOfResiduals = imageprocLib.calculateMaskedImageResiduals(differenceImage, badPixelMask)
    mwiu.Trace('lsst.imageproc.imageSubtract', 3, 'Mean and variance of residuals in difference image : %.3f %.3f (%d pixels)' %
               (meanOfResiduals, varianceOfResiduals, nGoodPixels))
    
    if debugIO:
        differenceImage.writeFits('diFits')

    return (differenceImage, psfMatchKernelPtr, backgroundFunctionPtr)
