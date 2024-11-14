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

"""Support utilities for Measuring sources"""

# Export DipoleTestImage to expose fake image generating funcs
__all__ = ["DipoleTestImage"]


import numpy as np
import lsst.geom as geom
import lsst.afw.detection as afwDet
import lsst.afw.display as afwDisplay
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg
import lsst.meas.base as measBase
from lsst.meas.algorithms.testUtils import plantSources
from lsst.ip.diffim.dipoleFitTask import DipoleFitAlgorithm
from lsst.ip.diffim import diffimLib

afwDisplay.setDefaultMaskTransparency(75)
keptPlots = False  # Have we arranged to keep spatial plots open?


def showSourceSet(sSet, xy0=(0, 0), frame=0, ctype=afwDisplay.GREEN, symb="+", size=2):
    """Draw the (XAstrom, YAstrom) positions of a set of Sources.

    Image has the given XY0.
    """
    disp = afwDisplay.afwDisplay(frame=frame)
    with disp.Buffering():
        for s in sSet:
            xc, yc = s.getXAstrom() - xy0[0], s.getYAstrom() - xy0[1]

            if symb == "id":
                disp.dot(str(s.getId()), xc, yc, ctype=ctype, size=size)
            else:
                disp.dot(symb, xc, yc, ctype=ctype, size=size)


# Kernel display utilities
#


def showKernelSpatialCells(maskedIm, kernelCellSet, showChi2=False, symb="o",
                           ctype=None, ctypeUnused=None, ctypeBad=None, size=3,
                           frame=None, title="Spatial Cells"):
    """Show the SpatialCells.

    If symb is something that display.dot understands (e.g. "o"), the top
    nMaxPerCell candidates will be indicated with that symbol, using ctype
    and size.
    """
    disp = afwDisplay.Display(frame=frame)
    disp.mtv(maskedIm, title=title)
    with disp.Buffering():
        origin = [-maskedIm.getX0(), -maskedIm.getY0()]
        for cell in kernelCellSet.getCellList():
            afwDisplay.utils.drawBBox(cell.getBBox(), origin=origin, display=disp)

            goodies = ctypeBad is None
            for cand in cell.begin(goodies):
                xc, yc = cand.getXCenter() + origin[0], cand.getYCenter() + origin[1]
                if cand.getStatus() == afwMath.SpatialCellCandidate.BAD:
                    color = ctypeBad
                elif cand.getStatus() == afwMath.SpatialCellCandidate.GOOD:
                    color = ctype
                elif cand.getStatus() == afwMath.SpatialCellCandidate.UNKNOWN:
                    color = ctypeUnused
                else:
                    continue

                if color:
                    disp.dot(symb, xc, yc, ctype=color, size=size)

                    if showChi2:
                        rchi2 = cand.getChi2()
                        if rchi2 > 1e100:
                            rchi2 = np.nan
                        disp.dot("%d %.1f" % (cand.getId(), rchi2),
                                 xc - size, yc - size - 4, ctype=color, size=size)


def showDiaSources(sources, exposure, isFlagged, isDipole, frame=None):
    """Display Dia Sources.
    """
    #
    # Show us the ccandidates
    #
    # Too many mask planes in diffims
    disp = afwDisplay.Display(frame=frame)
    for plane in ("BAD", "CR", "EDGE", "INTERPOlATED", "INTRP", "SAT", "SATURATED"):
        disp.setMaskPlaneColor(plane, color="ignore")

    mos = afwDisplay.utils.Mosaic()
    for i in range(len(sources)):
        source = sources[i]
        badFlag = isFlagged[i]
        dipoleFlag = isDipole[i]
        bbox = source.getFootprint().getBBox()
        stamp = exposure.Factory(exposure, bbox, True)
        im = afwDisplay.utils.Mosaic(gutter=1, background=0, mode="x")
        im.append(stamp.getMaskedImage())
        lab = "%.1f,%.1f:" % (source.getX(), source.getY())
        if badFlag:
            ctype = afwDisplay.RED
            lab += "BAD"
        if dipoleFlag:
            ctype = afwDisplay.YELLOW
            lab += "DIPOLE"
        if not badFlag and not dipoleFlag:
            ctype = afwDisplay.GREEN
            lab += "OK"
        mos.append(im.makeMosaic(), lab, ctype)
    title = "Dia Sources"
    mosaicImage = mos.makeMosaic(display=disp, title=title)
    return mosaicImage


def showKernelCandidates(kernelCellSet, kernel, background, frame=None, showBadCandidates=True,
                         resids=False, kernels=False):
    """Display the Kernel candidates.

    If kernel is provided include spatial model and residuals;
    If chi is True, generate a plot of residuals/sqrt(variance), i.e. chi.
    """
    #
    # Show us the ccandidates
    #
    if kernels:
        mos = afwDisplay.utils.Mosaic(gutter=5, background=0)
    else:
        mos = afwDisplay.utils.Mosaic(gutter=5, background=-1)
    #
    candidateCenters = []
    candidateCentersBad = []
    candidateIndex = 0
    for cell in kernelCellSet.getCellList():
        for cand in cell.begin(False):  # include bad candidates
            # Original difference image; if does not exist, skip candidate
            try:
                resid = cand.getDifferenceImage(diffimLib.KernelCandidateF.ORIG)
            except Exception:
                continue

            rchi2 = cand.getChi2()
            if rchi2 > 1e100:
                rchi2 = np.nan

            if not showBadCandidates and cand.isBad():
                continue

            im_resid = afwDisplay.utils.Mosaic(gutter=1, background=-0.5, mode="x")

            try:
                im = cand.getScienceMaskedImage()
                im = im.Factory(im, True)
                im.setXY0(cand.getScienceMaskedImage().getXY0())
            except Exception:
                continue
            if (not resids and not kernels):
                im_resid.append(im.Factory(im, True))
            try:
                im = cand.getTemplateMaskedImage()
                im = im.Factory(im, True)
                im.setXY0(cand.getTemplateMaskedImage().getXY0())
            except Exception:
                continue
            if (not resids and not kernels):
                im_resid.append(im.Factory(im, True))

            # Difference image with original basis
            if resids:
                var = resid.variance
                var = var.Factory(var, True)
                np.sqrt(var.array, var.array)  # inplace sqrt
                resid = resid.image
                resid /= var
                bbox = kernel.shrinkBBox(resid.getBBox())
                resid = resid.Factory(resid, bbox, deep=True)
            elif kernels:
                kim = cand.getKernelImage(diffimLib.KernelCandidateF.ORIG).convertF()
                resid = kim.Factory(kim, True)
            im_resid.append(resid)

            # residuals using spatial model
            ski = afwImage.ImageD(kernel.getDimensions())
            kernel.computeImage(ski, False, int(cand.getXCenter()), int(cand.getYCenter()))
            sk = afwMath.FixedKernel(ski)
            sbg = 0.0
            if background:
                sbg = background(int(cand.getXCenter()), int(cand.getYCenter()))
            sresid = cand.getDifferenceImage(sk, sbg)
            resid = sresid
            if resids:
                resid = sresid.image
                resid /= var
                bbox = kernel.shrinkBBox(resid.getBBox())
                resid = resid.Factory(resid, bbox, deep=True)
            elif kernels:
                kim = ski.convertF()
                resid = kim.Factory(kim, True)
            im_resid.append(resid)

            im = im_resid.makeMosaic()

            lab = "%d chi^2 %.1f" % (cand.getId(), rchi2)
            ctype = afwDisplay.RED if cand.isBad() else afwDisplay.GREEN

            mos.append(im, lab, ctype)

            if False and np.isnan(rchi2):
                disp = afwDisplay.Display(frame=1)
                disp.mtv(cand.getScienceMaskedImage.image, title="candidate")
                print("rating", cand.getCandidateRating())

            im = cand.getScienceMaskedImage()
            center = (candidateIndex, cand.getXCenter() - im.getX0(), cand.getYCenter() - im.getY0())
            candidateIndex += 1
            if cand.isBad():
                candidateCentersBad.append(center)
            else:
                candidateCenters.append(center)

    if resids:
        title = "chi Diffim"
    elif kernels:
        title = "Kernels"
    else:
        title = "Candidates & residuals"

    disp = afwDisplay.Display(frame=frame)
    mosaicImage = mos.makeMosaic(display=disp, title=title)

    return mosaicImage


def showKernelBasis(kernel, frame=None):
    """Display a Kernel's basis images.
    """
    mos = afwDisplay.utils.Mosaic()

    for k in kernel.getKernelList():
        im = afwImage.ImageD(k.getDimensions())
        k.computeImage(im, False)
        mos.append(im)

    disp = afwDisplay.Display(frame=frame)
    mos.makeMosaic(display=disp, title="Kernel Basis Images")

    return mos

###############


def plotKernelSpatialModel(kernel, kernelCellSet, showBadCandidates=True,
                           numSample=128, keepPlots=True, maxCoeff=10):
    """Plot the Kernel spatial model.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors
    except ImportError as e:
        print("Unable to import numpy and matplotlib: %s" % e)
        return

    x0 = kernelCellSet.getBBox().getBeginX()
    y0 = kernelCellSet.getBBox().getBeginY()

    candPos = list()
    candFits = list()
    badPos = list()
    badFits = list()
    candAmps = list()
    badAmps = list()
    for cell in kernelCellSet.getCellList():
        for cand in cell.begin(False):
            if not showBadCandidates and cand.isBad():
                continue
            candCenter = geom.PointD(cand.getXCenter(), cand.getYCenter())
            try:
                im = cand.getTemplateMaskedImage()
            except Exception:
                continue

            targetFits = badFits if cand.isBad() else candFits
            targetPos = badPos if cand.isBad() else candPos
            targetAmps = badAmps if cand.isBad() else candAmps

            # compare original and spatial kernel coefficients
            kp0 = np.array(cand.getKernel(diffimLib.KernelCandidateF.ORIG).getKernelParameters())
            amp = cand.getCandidateRating()

            targetFits = badFits if cand.isBad() else candFits
            targetPos = badPos if cand.isBad() else candPos
            targetAmps = badAmps if cand.isBad() else candAmps

            targetFits.append(kp0)
            targetPos.append(candCenter)
            targetAmps.append(amp)

    xGood = np.array([pos.getX() for pos in candPos]) - x0
    yGood = np.array([pos.getY() for pos in candPos]) - y0
    zGood = np.array(candFits)

    xBad = np.array([pos.getX() for pos in badPos]) - x0
    yBad = np.array([pos.getY() for pos in badPos]) - y0
    zBad = np.array(badFits)
    numBad = len(badPos)

    xRange = np.linspace(0, kernelCellSet.getBBox().getWidth(), num=numSample)
    yRange = np.linspace(0, kernelCellSet.getBBox().getHeight(), num=numSample)

    if maxCoeff:
        maxCoeff = min(maxCoeff, kernel.getNKernelParameters())
    else:
        maxCoeff = kernel.getNKernelParameters()

    for k in range(maxCoeff):
        func = kernel.getSpatialFunction(k)
        dfGood = zGood[:, k] - np.array([func(pos.getX(), pos.getY()) for pos in candPos])
        yMin = dfGood.min()
        yMax = dfGood.max()
        if numBad > 0:
            dfBad = zBad[:, k] - np.array([func(pos.getX(), pos.getY()) for pos in badPos])
            # Can really screw up the range...
            yMin = min([yMin, dfBad.min()])
            yMax = max([yMax, dfBad.max()])
        yMin -= 0.05*(yMax - yMin)
        yMax += 0.05*(yMax - yMin)

        fRange = np.ndarray((len(xRange), len(yRange)))
        for j, yVal in enumerate(yRange):
            for i, xVal in enumerate(xRange):
                fRange[j][i] = func(xVal, yVal)

        fig = plt.figure(k)

        fig.clf()
        try:
            fig.canvas._tkcanvas._root().lift()  # == Tk's raise, but raise is a python reserved word
        except Exception:                                 # protect against API changes
            pass

        fig.suptitle('Kernel component %d' % k)

        # LL
        ax = fig.add_axes((0.1, 0.05, 0.35, 0.35))
        vmin = fRange.min()  # - 0.05*np.fabs(fRange.min())
        vmax = fRange.max()  # + 0.05*np.fabs(fRange.max())
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(fRange, aspect='auto', norm=norm,
                       extent=[0, kernelCellSet.getBBox().getWidth() - 1,
                               0, kernelCellSet.getBBox().getHeight() - 1])
        ax.set_title('Spatial polynomial')
        plt.colorbar(im, orientation='horizontal', ticks=[vmin, vmax])

        # UL
        ax = fig.add_axes((0.1, 0.55, 0.35, 0.35))
        ax.plot(-2.5*np.log10(candAmps), zGood[:, k], 'b+')
        if numBad > 0:
            ax.plot(-2.5*np.log10(badAmps), zBad[:, k], 'r+')
        ax.set_title("Basis Coefficients")
        ax.set_xlabel("Instr mag")
        ax.set_ylabel("Coeff")

        # LR
        ax = fig.add_axes((0.55, 0.05, 0.35, 0.35))
        ax.set_autoscale_on(False)
        ax.set_xbound(lower=0, upper=kernelCellSet.getBBox().getHeight())
        ax.set_ybound(lower=yMin, upper=yMax)
        ax.plot(yGood, dfGood, 'b+')
        if numBad > 0:
            ax.plot(yBad, dfBad, 'r+')
        ax.axhline(0.0)
        ax.set_title('dCoeff (indiv-spatial) vs. y')

        # UR
        ax = fig.add_axes((0.55, 0.55, 0.35, 0.35))
        ax.set_autoscale_on(False)
        ax.set_xbound(lower=0, upper=kernelCellSet.getBBox().getWidth())
        ax.set_ybound(lower=yMin, upper=yMax)
        ax.plot(xGood, dfGood, 'b+')
        if numBad > 0:
            ax.plot(xBad, dfBad, 'r+')
        ax.axhline(0.0)
        ax.set_title('dCoeff (indiv-spatial) vs. x')

        fig.show()

    global keptPlots
    if keepPlots and not keptPlots:
        # Keep plots open when done
        def show():
            print("%s: Please close plots when done." % __name__)
            try:
                plt.show()
            except Exception:
                pass
            print("Plots closed, exiting...")
        import atexit
        atexit.register(show)
        keptPlots = True


def plotKernelCoefficients(spatialKernel, kernelCellSet, showBadCandidates=False, keepPlots=True):
    """Plot the individual kernel candidate and the spatial kernel solution coefficients.

    Parameters
    ----------

    spatialKernel : `lsst.afw.math.LinearCombinationKernel`
        The spatial spatialKernel solution model which is a spatially varying linear combination
        of the spatialKernel basis functions.
        Typically returned by `lsst.ip.diffim.SpatialKernelSolution.getSolutionPair()`.

    kernelCellSet : `lsst.afw.math.SpatialCellSet`
        The spatial cells that was used for solution for the spatialKernel. They contain the
        local solutions of the AL kernel for the selected sources.

    showBadCandidates : `bool`, optional
        If True, plot the coefficient values for kernel candidates where the solution was marked
        bad by the numerical algorithm. Defaults to False.

    keepPlots: `bool`, optional
        If True, sets ``plt.show()`` to be called before the task terminates, so that the plots
        can be explored interactively. Defaults to True.

    Notes
    -----
    This function produces 3 figures per image subtraction operation.
    * A grid plot of the local solutions. Each grid cell corresponds to a proportional area in
      the image. In each cell, local kernel solution coefficients are plotted of kernel candidates (color)
      that fall into this area as a function of the kernel basis function number.
    * A grid plot of the spatial solution. Each grid cell corresponds to a proportional area in
      the image. In each cell, the spatial solution coefficients are evaluated for the center of the cell.
    * Histogram of the local solution coefficients. Red line marks the spatial solution value at
      center of the image.

    This function is called if ``lsst.ip.diffim.psfMatch.plotKernelCoefficients==True`` in lsstDebug. This
    function was implemented as part of DM-17825.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("Unable to import matplotlib: %s" % e)
        return

    # Image dimensions
    imgBBox = kernelCellSet.getBBox()
    x0 = imgBBox.getBeginX()
    y0 = imgBBox.getBeginY()
    wImage = imgBBox.getWidth()
    hImage = imgBBox.getHeight()
    imgCenterX = imgBBox.getCenterX()
    imgCenterY = imgBBox.getCenterY()

    # Plot the local solutions
    # ----

    # Grid size
    nX = 8
    nY = 8
    wCell = wImage / nX
    hCell = hImage / nY

    fig = plt.figure()
    fig.suptitle("Kernel candidate parameters on an image grid")
    arrAx = fig.subplots(nrows=nY, ncols=nX, sharex=True, sharey=True, gridspec_kw=dict(
        wspace=0, hspace=0))

    # Bottom left panel is for bottom left part of the image
    arrAx = arrAx[::-1, :]

    allParams = []
    for cell in kernelCellSet.getCellList():
        cellBBox = geom.Box2D(cell.getBBox())
        # Determine which panel this spatial cell belongs to
        iX = int((cellBBox.getCenterX() - x0)//wCell)
        iY = int((cellBBox.getCenterY() - y0)//hCell)

        for cand in cell.begin(False):
            try:
                kernel = cand.getKernel(cand.ORIG)
            except Exception:
                continue

            if not showBadCandidates and cand.isBad():
                continue

            nKernelParams = kernel.getNKernelParameters()
            kernelParams = np.array(kernel.getKernelParameters())
            allParams.append(kernelParams)

            if cand.isBad():
                color = 'red'
            else:
                color = None
            arrAx[iY, iX].plot(np.arange(nKernelParams), kernelParams, '.-',
                               color=color, drawstyle='steps-mid', linewidth=0.1)
    for ax in arrAx.ravel():
        ax.grid(True, axis='y')

    # Plot histogram of the local parameters and the global solution at the image center
    # ----

    spatialFuncs = spatialKernel.getSpatialFunctionList()
    nKernelParams = spatialKernel.getNKernelParameters()
    nX = 8
    fig = plt.figure()
    fig.suptitle("Hist. of parameters marked with spatial solution at img center")
    arrAx = fig.subplots(nrows=int(nKernelParams//nX)+1, ncols=nX)
    arrAx = arrAx[::-1, :]
    allParams = np.array(allParams)
    for k in range(nKernelParams):
        ax = arrAx.ravel()[k]
        ax.hist(allParams[:, k], bins=20, edgecolor='black')
        ax.set_xlabel('P{}'.format(k))
        valueParam = spatialFuncs[k](imgCenterX, imgCenterY)
        ax.axvline(x=valueParam, color='red')
        ax.text(0.1, 0.9, '{:.1f}'.format(valueParam),
                transform=ax.transAxes, backgroundcolor='lightsteelblue')

    # Plot grid of the spatial solution
    # ----

    nX = 8
    nY = 8
    wCell = wImage / nX
    hCell = hImage / nY
    x0 += wCell / 2
    y0 += hCell / 2

    fig = plt.figure()
    fig.suptitle("Spatial solution of kernel parameters on an image grid")
    arrAx = fig.subplots(nrows=nY, ncols=nX, sharex=True, sharey=True, gridspec_kw=dict(
        wspace=0, hspace=0))
    arrAx = arrAx[::-1, :]
    kernelParams = np.zeros(nKernelParams, dtype=float)

    for iX in range(nX):
        for iY in range(nY):
            x = x0 + iX * wCell
            y = y0 + iY * hCell
            # Evaluate the spatial solution functions for this x,y location
            kernelParams = [f(x, y) for f in spatialFuncs]
            arrAx[iY, iX].plot(np.arange(nKernelParams), kernelParams, '.-', drawstyle='steps-mid')
            arrAx[iY, iX].grid(True, axis='y')

    global keptPlots
    if keepPlots and not keptPlots:
        # Keep plots open when done
        def show():
            print("%s: Please close plots when done." % __name__)
            try:
                plt.show()
            except Exception:
                pass
            print("Plots closed, exiting...")
        import atexit
        atexit.register(show)
        keptPlots = True


def showKernelMosaic(bbox, kernel, nx=7, ny=None, frame=None, title=None,
                     showCenter=True, showEllipticity=True):
    """Show a mosaic of Kernel images.
    """
    mos = afwDisplay.utils.Mosaic()

    x0 = bbox.getBeginX()
    y0 = bbox.getBeginY()
    width = bbox.getWidth()
    height = bbox.getHeight()

    if not ny:
        ny = int(nx*float(height)/width + 0.5)
        if not ny:
            ny = 1

    schema = afwTable.SourceTable.makeMinimalSchema()
    centroidName = "base_SdssCentroid"
    shapeName = "base_SdssShape"
    control = measBase.SdssCentroidControl()
    schema.getAliasMap().set("slot_Centroid", centroidName)
    schema.getAliasMap().set("slot_Centroid_flag", centroidName + "_flag")
    centroider = measBase.SdssCentroidAlgorithm(control, centroidName, schema)
    sdssShape = measBase.SdssShapeControl()
    shaper = measBase.SdssShapeAlgorithm(sdssShape, shapeName, schema)
    table = afwTable.SourceTable.make(schema)
    table.defineCentroid(centroidName)
    table.defineShape(shapeName)

    centers = []
    shapes = []
    for iy in range(ny):
        for ix in range(nx):
            x = int(ix*(width - 1)/(nx - 1)) + x0
            y = int(iy*(height - 1)/(ny - 1)) + y0

            im = afwImage.ImageD(kernel.getDimensions())
            ksum = kernel.computeImage(im, False, x, y)
            lab = "Kernel(%d,%d)=%.2f" % (x, y, ksum) if False else ""
            mos.append(im, lab)

            # SdssCentroidAlgorithm.measure requires an exposure of floats
            exp = afwImage.makeExposure(afwImage.makeMaskedImage(im.convertF()))

            w, h = im.getWidth(), im.getHeight()
            centerX = im.getX0() + w//2
            centerY = im.getY0() + h//2
            src = table.makeRecord()
            spans = afwGeom.SpanSet(exp.getBBox())
            foot = afwDet.Footprint(spans)
            foot.addPeak(centerX, centerY, 1)
            src.setFootprint(foot)

            try:  # The centroider requires a psf, so this will fail if none is attached to exp
                centroider.measure(src, exp)
                centers.append((src.getX(), src.getY()))

                shaper.measure(src, exp)
                shapes.append((src.getIxx(), src.getIxy(), src.getIyy()))
            except Exception:
                pass

    disp = afwDisplay.Display(frame=frame)
    mos.makeMosaic(display=disp, title=title if title else "Model Kernel", mode=nx)

    if centers and frame is not None:
        disp = afwDisplay.Display(frame=frame)
        i = 0
        with disp.Buffering():
            for cen, shape in zip(centers, shapes):
                bbox = mos.getBBox(i)
                i += 1
                xc, yc = cen[0] + bbox.getMinX(), cen[1] + bbox.getMinY()
                if showCenter:
                    disp.dot("+", xc, yc, ctype=afwDisplay.BLUE)

                if showEllipticity:
                    ixx, ixy, iyy = shape
                    disp.dot("@:%g,%g,%g" % (ixx, ixy, iyy), xc, yc, ctype=afwDisplay.RED)

    return mos


def plotWhisker(results, newWcs):
    """Plot whisker diagram of astromeric offsets between results.matches.
    """
    refCoordKey = results.matches[0].first.getTable().getCoordKey()
    inCentroidKey = results.matches[0].second.getTable().getCentroidSlot().getMeasKey()
    positions = [m.first.get(refCoordKey) for m in results.matches]
    residuals = [m.first.get(refCoordKey).getOffsetFrom(
        newWcs.pixelToSky(m.second.get(inCentroidKey))) for
        m in results.matches]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    sp = fig.add_subplot(1, 1, 0)
    xpos = [x[0].asDegrees() for x in positions]
    ypos = [x[1].asDegrees() for x in positions]
    xpos.append(0.02*(max(xpos) - min(xpos)) + min(xpos))
    ypos.append(0.98*(max(ypos) - min(ypos)) + min(ypos))
    xidxs = np.isfinite(xpos)
    yidxs = np.isfinite(ypos)
    X = np.asarray(xpos)[xidxs]
    Y = np.asarray(ypos)[yidxs]
    distance = [x[1].asArcseconds() for x in residuals]
    distance.append(0.2)
    distance = np.asarray(distance)[xidxs]
    # NOTE: This assumes that the bearing is measured positive from +RA through North.
    # From the documentation this is not clear.
    bearing = [x[0].asRadians() for x in residuals]
    bearing.append(0)
    bearing = np.asarray(bearing)[xidxs]
    U = (distance*np.cos(bearing))
    V = (distance*np.sin(bearing))
    sp.quiver(X, Y, U, V)
    sp.set_title("WCS Residual")
    plt.show()


class DipoleTestImage:
    """Utility class for dipole measurement testing.

    Generate an image with simulated dipoles and noise; store the original
    "pre-subtraction" images and catalogs as well.
    Used to generate test data for DMTN-007 (http://dmtn-007.lsst.io).
    """

    def __init__(self, w=101, h=101, xcenPos=[27.], ycenPos=[25.], xcenNeg=[23.], ycenNeg=[25.],
                 psfSigma=2., flux=[30000.], fluxNeg=None, noise=10., gradientParams=None, edgeWidth=8):
        self.w = w
        self.h = h
        self.xcenPos = xcenPos
        self.ycenPos = ycenPos
        self.xcenNeg = xcenNeg
        self.ycenNeg = ycenNeg
        self.psfSigma = psfSigma
        self.flux = flux
        self.fluxNeg = fluxNeg
        if fluxNeg is None:
            self.fluxNeg = self.flux
        self.noise = noise
        self.gradientParams = gradientParams
        self.edgeWidth = edgeWidth
        self._makeDipoleImage()

    def _makeDipoleImage(self):
        """Generate an exposure and catalog with the given dipole source(s).
        """
        # Must seed the pos/neg images with different values to ensure they get different noise realizations
        posImage, posCatalog = self._makeStarImage(
            xc=self.xcenPos, yc=self.ycenPos, flux=self.flux, randomSeed=111)

        negImage, negCatalog = self._makeStarImage(
            xc=self.xcenNeg, yc=self.ycenNeg, flux=self.fluxNeg, randomSeed=222)

        dipole = posImage.clone()
        di = dipole.getMaskedImage()
        di -= negImage.getMaskedImage()

        self.diffim, self.posImage, self.posCatalog, self.negImage, self.negCatalog \
            = dipole, posImage, posCatalog, negImage, negCatalog

    def _makeStarImage(self, xc=[15.3], yc=[18.6], flux=[2500], schema=None, randomSeed=None):
        """Generate an exposure and catalog with the given stellar source(s).
        """
        from lsst.meas.base.tests import TestDataset
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Point2I(self.w - 1, self.h - 1))
        dataset = TestDataset(bbox, psfSigma=self.psfSigma, threshold=1.)

        for i in range(len(xc)):
            dataset.addSource(instFlux=flux[i], centroid=geom.Point2D(xc[i], yc[i]))

        if schema is None:
            schema = TestDataset.makeMinimalSchema()
        exposure, catalog = dataset.realize(noise=self.noise, schema=schema, randomSeed=randomSeed)

        # set EDGE by masking the whole exposure and un-masking an inner bbox
        edgeMask = exposure.mask.getPlaneBitMask('EDGE')
        exposure.mask.array |= edgeMask
        inner_bbox = exposure.getBBox()
        inner_bbox.grow(-self.edgeWidth)
        exposure[inner_bbox].mask.array &= ~edgeMask

        if self.gradientParams is not None:
            y, x = np.mgrid[:self.w, :self.h]
            gp = self.gradientParams
            gradient = gp[0] + gp[1]*x + gp[2]*y
            if len(self.gradientParams) > 3:  # it includes a set of 2nd-order polynomial params
                gradient += gp[3]*x*y + gp[4]*x*x + gp[5]*y*y
            imgArr = exposure.image.array
            imgArr += gradient

        return exposure, catalog

    def fitDipoleSource(self, source, **kwds):
        alg = DipoleFitAlgorithm(self.diffim, self.posImage, self.negImage)
        fitResult = alg.fitDipole(source, **kwds)
        return fitResult

    def detectDipoleSources(self, doMerge=True, diffim=None, detectSigma=5.5, grow=3, minBinSize=32):
        """Utility function for detecting dipoles.

        Detect pos/neg sources in the diffim, then merge them. A
        bigger "grow" parameter leads to a larger footprint which
        helps with dipole measurement for faint dipoles.

        Parameters
        ----------
        doMerge : `bool`
           Whether to merge the positive and negagive detections into a single
           source table.
        diffim : `lsst.afw.image.exposure.exposure.ExposureF`
           Difference image on which to perform detection.
        detectSigma : `float`
           Threshold for object detection.
        grow : `int`
           Number of pixels to grow the footprints before merging.
        minBinSize : `int`
           Minimum bin size for the background (re)estimation (only applies if
           the default leads to min(nBinX, nBinY) < fit order so the default
           config parameter needs to be decreased, but not to a value smaller
           than ``minBinSize``, in which case the fitting algorithm will take
           over and decrease the fit order appropriately.)

        Returns
        -------
        sources : `lsst.afw.table.SourceCatalog`
           If doMerge=True, the merged source catalog is returned OR
        detectTask : `lsst.meas.algorithms.SourceDetectionTask`
        schema : `lsst.afw.table.Schema`
           If doMerge=False, the source detection task and its schema are
           returned.
        """
        if diffim is None:
            diffim = self.diffim

        # Start with a minimal schema - only the fields all SourceCatalogs need
        schema = afwTable.SourceTable.makeMinimalSchema()

        # Customize the detection task a bit (optional)
        detectConfig = measAlg.SourceDetectionConfig()
        detectConfig.returnOriginalFootprints = False  # should be the default

        # code from imageDifference.py:
        detectConfig.thresholdPolarity = "both"
        detectConfig.thresholdValue = detectSigma
        # detectConfig.nSigmaToGrow = psfSigma
        detectConfig.reEstimateBackground = True  # if False, will fail often for faint sources on gradients?
        detectConfig.thresholdType = "pixel_stdev"
        detectConfig.excludeMaskPlanes = ["EDGE"]
        # Test images are often quite small, so may need to adjust background binSize
        while ((min(diffim.getWidth(), diffim.getHeight()))//detectConfig.background.binSize
               < detectConfig.background.approxOrderX and detectConfig.background.binSize > minBinSize):
            detectConfig.background.binSize = max(minBinSize, detectConfig.background.binSize//2)

        # Create the detection task. We pass the schema so the task can declare a few flag fields
        detectTask = measAlg.SourceDetectionTask(schema, config=detectConfig)

        table = afwTable.SourceTable.make(schema)
        catalog = detectTask.run(table, diffim)

        # Now do the merge.
        if doMerge:
            fpSet = catalog.positive
            fpSet.merge(catalog.negative, grow, grow, False)
            sources = afwTable.SourceCatalog(table)
            fpSet.makeSources(sources)

            return sources

        else:
            return detectTask, schema


def detectTestSources(exposure, addMaskPlanes=None):
    """Minimal source detection wrapper suitable for unit tests.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Exposure on which to run detection/measurement
        The exposure is modified in place to set the 'DETECTED' mask plane.
    addMaskPlanes : `list` of `str`, optional
        Additional mask planes to add to the maskedImage of the exposure.

    Returns
    -------
    selectSources
        Source catalog containing candidates
    """
    if addMaskPlanes is None:
        # And add empty INJECTED and INJECTED_TEMPLATE mask planes
        addMaskPlanes = ["INJECTED", "INJECTED_TEMPLATE"]

    schema = afwTable.SourceTable.makeMinimalSchema()

    # Add coordinate error fields:
    afwTable.CoordKey.addErrorFields(schema)
    selectDetection = measAlg.SourceDetectionTask(schema=schema)
    selectMeasurement = measBase.SingleFrameMeasurementTask(schema=schema)
    table = afwTable.SourceTable.make(schema)

    detRet = selectDetection.run(
        table=table,
        exposure=exposure,
        sigma=None,  # The appropriate sigma is calculated from the PSF
        doSmooth=True
    )
    for mp in addMaskPlanes:
        exposure.mask.addMaskPlane(mp)

    selectSources = detRet.sources
    selectMeasurement.run(measCat=selectSources, exposure=exposure)

    return selectSources


def makeFakeWcs():
    """Make a fake, affine Wcs.
    """
    crpix = geom.Point2D(123.45, 678.9)
    crval = geom.SpherePoint(0.1, 0.1, geom.degrees)
    cdMatrix = np.array([[5.19513851e-05, -2.81124812e-07],
                        [-3.25186974e-07, -5.19112119e-05]])
    return afwGeom.makeSkyWcs(crpix, crval, cdMatrix)


def makeTestImage(seed=5, nSrc=20, psfSize=2., noiseLevel=5.,
                  noiseSeed=6, fluxLevel=500., fluxRange=2.,
                  kernelSize=32, templateBorderSize=0,
                  background=None,
                  xSize=256,
                  ySize=256,
                  x0=12345,
                  y0=67890,
                  calibration=1.,
                  doApplyCalibration=False,
                  xLoc=None,
                  yLoc=None,
                  flux=None,
                  clearEdgeMask=False,
                  addMaskPlanes=None,
                  band="g",
                  physicalFilter="g NotACamera"
                  ):
    """Make a reproduceable PSF-convolved exposure for testing.

    Parameters
    ----------
    seed : `int`, optional
        Seed value to initialize the random number generator for sources.
    nSrc : `int`, optional
        Number of sources to simulate.
    psfSize : `float`, optional
        Width of the PSF of the simulated sources, in pixels.
    noiseLevel : `float`, optional
        Standard deviation of the noise to add to each pixel.
    noiseSeed : `int`, optional
        Seed value to initialize the random number generator for noise.
    fluxLevel : `float`, optional
        Reference flux of the simulated sources.
    fluxRange : `float`, optional
        Range in flux amplitude of the simulated sources.
    kernelSize : `int`, optional
        Size in pixels of the kernel for simulating sources.
    templateBorderSize : `int`, optional
        Size in pixels of the image border used to pad the image.
    background : `lsst.afw.math.Chebyshev1Function2D`, optional
        Optional background to add to the output image.
    xSize, ySize : `int`, optional
        Size in pixels of the simulated image.
    x0, y0 : `int`, optional
        Origin of the image.
    calibration : `float`, optional
        Conversion factor between instFlux and nJy.
    doApplyCalibration : `bool`, optional
        Apply the photometric calibration and return the image in nJy?
    xLoc, yLoc : `list` of `float`, optional
        User-specified coordinates of the simulated sources.
        If specified, must have length equal to ``nSrc``
    flux : `list` of `float`, optional
        User-specified fluxes of the simulated sources.
        If specified, must have length equal to ``nSrc``
    clearEdgeMask : `bool`, optional
        Clear the "EDGE" mask plane after source detection.
    addMaskPlanes : `list` of `str`, optional
        Mask plane names to add to the image.

    Returns
    -------
    modelExposure : `lsst.afw.image.Exposure`
        The model image, with the mask and variance planes. The DETECTED
        plane is filled in for the injected source footprints.
    sourceCat : `lsst.afw.table.SourceCatalog`
        Catalog of sources inserted in the model image.

    Raises
    ------
    ValueError
        If `xloc`, `yloc`, or `flux` are supplied with inconsistant lengths.
    """
    # Distance from the inner edge of the bounding box to avoid placing test
    # sources in the model images.
    bufferSize = kernelSize/2 + templateBorderSize + 1

    bbox = geom.Box2I(geom.Point2I(x0, y0), geom.Extent2I(xSize, ySize))
    if templateBorderSize > 0:
        bbox.grow(templateBorderSize)

    rng = np.random.RandomState(seed)
    rngNoise = np.random.RandomState(noiseSeed)
    x0, y0 = bbox.getBegin()
    xSize, ySize = bbox.getDimensions()
    if xLoc is None:
        xLoc = rng.rand(nSrc)*(xSize - 2*bufferSize) + bufferSize + x0
    else:
        if len(xLoc) != nSrc:
            raise ValueError("xLoc must have length equal to nSrc. %f supplied vs %f", len(xLoc), nSrc)
    if yLoc is None:
        yLoc = rng.rand(nSrc)*(ySize - 2*bufferSize) + bufferSize + y0
    else:
        if len(yLoc) != nSrc:
            raise ValueError("yLoc must have length equal to nSrc. %f supplied vs %f", len(yLoc), nSrc)

    if flux is None:
        flux = (rng.rand(nSrc)*(fluxRange - 1.) + 1.)*fluxLevel
    else:
        if len(flux) != nSrc:
            raise ValueError("flux must have length equal to nSrc. %f supplied vs %f", len(flux), nSrc)
    sigmas = [psfSize for src in range(nSrc)]
    injectList = list(zip(xLoc, yLoc, flux, sigmas))
    skyLevel = 0
    # Don't use the built in poisson noise: it modifies the global state of numpy random
    modelExposure = plantSources(bbox, kernelSize, skyLevel, injectList, addPoissonNoise=False)
    modelExposure.setWcs(makeFakeWcs())
    noise = rngNoise.randn(ySize, xSize)*noiseLevel
    noise -= np.mean(noise)
    modelExposure.variance.array = np.sqrt(np.abs(modelExposure.image.array)) + noiseLevel**2
    modelExposure.image.array += noise

    # Run source detection to set up the mask plane
    detectTestSources(modelExposure, addMaskPlanes=addMaskPlanes)
    if clearEdgeMask:
        modelExposure.mask &= ~modelExposure.mask.getPlaneBitMask("EDGE")
    modelExposure.setPhotoCalib(afwImage.PhotoCalib(calibration, 0., bbox))
    if background is not None:
        modelExposure.image += background
    modelExposure.maskedImage /= calibration
    modelExposure.setFilter(afwImage.FilterLabel(band, physicalFilter))
    modelExposure.info.setId(seed)
    if doApplyCalibration:
        modelExposure.maskedImage = modelExposure.photoCalib.calibrateImage(modelExposure.maskedImage)

    truth = _fillTruthCatalog(injectList)

    return modelExposure, truth


def _makeTruthSchema():
    """Make a schema for the truth catalog produced by `makeTestImage`.

    Returns
    -------
    keys : `dict` [`str`]
        Fields added to the catalog, to make it easier to set them.
    schema : `lsst.afw.table.Schema`
        Schema to use to make a "truth" SourceCatalog.
        Calib, Ap, and Psf flux slots all are set to ``truth_instFlux``.
    """
    schema = afwTable.SourceTable.makeMinimalSchema()
    keys = {}
    # Don't use a FluxResultKey so we can manage the flux and err separately.
    keys["instFlux"] = schema.addField("truth_instFlux", type=np.float64,
                                       doc="true instFlux", units="count")
    keys["instFluxErr"] = schema.addField("truth_instFluxErr", type=np.float64,
                                          doc="true instFluxErr", units="count")
    keys["centroid"] = afwTable.Point2DKey.addFields(schema, "truth", "true simulated centroid", "pixel")
    schema.addField("truth_flag", "Flag", "truth flux failure flag.")
    # Add the flag fields a source selector would need.
    schema.addField("sky_source", "Flag", "testing flag.")
    schema.addField("base_PixelFlags_flag_interpolated", "Flag", "testing flag.")
    schema.addField("base_PixelFlags_flag_saturated", "Flag", "testing flag.")
    schema.addField("base_PixelFlags_flag_bad", "Flag", "testing flag.")
    schema.addField("base_PixelFlags_flag_edge", "Flag", "testing flag.")
    schema.addField("base_PixelFlags_flag_nodata", "Flag", "testing flag.")
    schema.addField("base_PsfFlux_flag", "Flag", "testing flag.")
    schema.addField("base_ClassificationSizeExtendedness_value", "Flag", "testing flag.")
    schema.addField("deblend_nChild", "Flag", "testing flag.")
    schema.addField("detect_isPrimary", "Flag", "testing flag.")
    schema.addField("calib_psf_used", "Flag", "testing flag.")
    schema.getAliasMap().set("slot_Centroid", "truth")
    schema.getAliasMap().set("slot_CalibFlux", "truth")
    schema.getAliasMap().set("slot_ApFlux", "truth")
    schema.getAliasMap().set("slot_PsfFlux", "truth")
    return keys, schema


def _fillTruthCatalog(injectList):
    """Add injected sources to the truth catalog.

    Parameters
    ----------
    injectList : `list` [`float`]
        Sources that were injected; tuples of (x, y, flux, size).

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        Catalog with centroids and instFlux/instFluxErr values filled in and
        appropriate slots set.
    """
    keys, schema = _makeTruthSchema()
    catalog = afwTable.SourceCatalog(schema)
    catalog.reserve(len(injectList))
    for x, y, flux, size in injectList:
        record = catalog.addNew()
        keys["centroid"].set(record, geom.PointD(x, y))
        keys["instFlux"].set(record, flux)
        # Approximate injected errors
        keys["instFluxErr"].set(record, 20)
        # 5-sigma effective source width
        circle = afwGeom.Ellipse(afwGeom.ellipses.Axes(5*size, 5*size, 0), geom.Point2D(x, y))
        footprint = afwDetection.Footprint(afwGeom.SpanSet.fromShape(circle))
        footprint.addPeak(x, y, flux)
        record.setFootprint(footprint)

        # Set source records for isolated stars
        record["base_ClassificationSizeExtendedness_value"] = 0
        record["deblend_nChild"] = 0
        record["detect_isPrimary"] = True
        record["calib_psf_used"] = True

    return catalog


def makeStats(badMaskPlanes=None):
    """Create a statistics control for configuring calculations on images.

    Parameters
    ----------
    badMaskPlanes : `list` of `str`, optional
        List of mask planes to exclude from calculations.

    Returns
    -------
    statsControl : ` lsst.afw.math.StatisticsControl`
        Statistics control object for configuring calculations on images.
    """
    if badMaskPlanes is None:
        badMaskPlanes = ("INTRP", "EDGE", "DETECTED", "SAT", "CR",
                         "BAD", "NO_DATA", "DETECTED_NEGATIVE")
    statsControl = afwMath.StatisticsControl()
    statsControl.setNumSigmaClip(3.)
    statsControl.setNumIter(3)
    statsControl.setAndMask(afwImage.Mask.getPlaneBitMask(badMaskPlanes))
    return statsControl


class CustomCoaddPsf(measAlg.CoaddPsf):
    """A custom CoaddPSF that overrides the getAveragePosition method.

    It intentionally moves the position off-image to cause a test failure.
    """
    def getAveragePosition(self):
        return geom.Point2D(-10000, -10000)
