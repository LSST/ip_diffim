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

import sys
import optparse

import lsst.afw.image as afwImage
import lsst.log.utils as logUtils

import lsst.ip.diffim as ipDiffim
import lsst.ip.diffim.diffimTools as diffimTools
import lsst.afw.display as afwDisplay


def main():
    defSciencePath = None
    defTemplatePath = None
    defOutputPath = 'diffImage.fits'
    defVerbosity = 5
    defFwhm = 3.5

    usage = """usage: %%prog [options] [scienceImage [templateImage [outputImage]]]]

Notes:
- image arguments are paths to MaskedImage fits files
- image arguments must NOT include the final _img.fits
- the result is science image - template image
- the template image is convolved, the science image is not
- default scienceMaskedImage=%s
- default templateMaskedImage=%s
- default outputImage=%s
""" % (defSciencePath, defTemplatePath, defOutputPath)

    parser = optparse.OptionParser(usage)
    parser.add_option('-v', '--verbosity', type=int, default=defVerbosity,
                      help='verbosity of Trace messages')
    parser.add_option('-d', '--display', action='store_true', default=False,
                      help='display the images')
    parser.add_option('-b', '--bg', action='store_true', default=False,
                      help='subtract backgrounds')
    parser.add_option('--fwhmS', type=float,
                      help='Science Image Psf Fwhm (pixel)')
    parser.add_option('--fwhmT', type=float,
                      help='Template Image Psf Fwhm (pixel)')
    parser.add_option('-w', '--writeOutput', action='store_true', default=False,
                      help='Write out the subtracted image?')

    (options, args) = parser.parse_args()

    def getArg(ind, defValue):
        if ind < len(args):
            return args[ind]
        return defValue

    sciencePath = getArg(0, defSciencePath)
    templatePath = getArg(1, defTemplatePath)
    outputPath = getArg(2, defOutputPath)

    if sciencePath is None or templatePath is None:
        parser.print_help()
        sys.exit(1)

    print('Science image: ', sciencePath)
    print('Template image:', templatePath)
    print('Output image:  ', outputPath)

    fwhmS = defFwhm
    if options.fwhmS:
        print('FwhmS =', options.fwhmS)
        fwhmS = options.fwhmS

    fwhmT = defFwhm
    if options.fwhmT:
        print('Fwhmt =', options.fwhmT)
        fwhmT = options.fwhmT

    display = False
    if options.display:
        print('Display =', options.display)
        display = True

    bgSub = False
    if options.bg:
        print('Background subtract =', options.bg)
        bgSub = True

    if options.verbosity > 0:
        print('Verbosity =', options.verbosity)
        logUtils.traceSetAt("lsst.ip.diffim", options.verbosity)

    writeOutput = False
    if options.writeOutput:
        print('writeOutput =', options.writeOutput)
        writeOutput = True

    ####

    templateExposure = afwImage.ExposureF(templatePath)
    scienceExposure = afwImage.ExposureF(sciencePath)
    templateMaskedImage = templateExposure.getMaskedImage()
    scienceMaskedImage = scienceExposure.getMaskedImage()

    config = ipDiffim.ImagePsfMatchTask.ConfigClass()
    config.kernel.name = "AL"
    subconfig = config.kernel.active

    if bgSub:
        diffimTools.backgroundSubtract(subconfig.afwBackgroundConfig,
                                       [templateMaskedImage, scienceMaskedImage])
    else:
        if not subconfig.fitForBackground:
            print('NOTE: no background subtraction at all is requested')

    psfmatch = ipDiffim.ImagePsfMatchTask(config)
    candidateList = psfmatch.makeCandidateList(templateExposure, scienceExposure, subconfig.kernelSize)
    results = psfmatch.subtractMaskedImages(templateMaskedImage, scienceMaskedImage, candidateList,
                                            templateFwhmPix=fwhmT, scienceFwhmPix=fwhmS)
    differenceMaskedImage = results.subtractedMaskedImage
    if writeOutput:
        differenceMaskedImage.writeFits(outputPath)

    if False:
        spatialKernel = results.psfMatchingKernel
        print(spatialKernel.getSpatialParameters())

    if display:
        afwDisplay.Display().mtv(differenceMaskedImage, title="Difference Masked Image")


def run():
    main()


if __name__ == '__main__':
    run()
