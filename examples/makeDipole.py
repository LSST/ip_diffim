import numpy as np
import lsst.afw.display.ds9 as ds9
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.ip.diffim as ipDiffim
import lsst.ip.diffim.diffimTools as diffimTools

w,h = 100,100
xc,yc=50,50
scaling = 100.0

# Make random noise image
image = afwImage.MaskedImageF(w,h)
image.set(0)
array = image.getImage().getArray()
array[:,:] = np.random.randn(w,h)
var   = image.getVariance()
var.set(1.0)
ds9.mtv(image, frame=1)
ds9.mtv(image.getVariance(), frame=2)
# Create Psf for dipole creation and measurement
psf = afwDet.createPsf("DoubleGaussian", 17, 17, 2.0, 3.5, 0.1)
#psf = afwDet.createPsf("DoubleGaussian", 11, 11, 1.0, 2.5, 0.1)
psfim = psf.computeImage(afwGeom.Point2D(0., 0.)).convertF()
psfw, psfh = psfim.getDimensions()
psfSum = np.sum(psfim.getArray())
print "ORIGINAL PSF SUM:", psfSum * scaling

# For the dipole
array = image.getImage().getArray()
xp, yp = xc-psfw+psfw//2, yc-psfh+psfh//2
array[yp:yp+psfh, xp:xp+psfw] += psfim.getArray() * scaling
xn, yn = xc-psfw//2, yc-psfh//2
array[yn:yn+psfh, xn:xn+psfw] -= psfim.getArray() * scaling
ds9.mtv(image, frame=3)

# Create an exposure, detect positive and negative peaks separately
exp = afwImage.makeExposure(image)
exp.setPsf(psf)
config = measAlg.SourceDetectionConfig()
config.thresholdPolarity = "both"
config.reEstimateBackground = False
schema = afwTable.SourceTable.makeMinimalSchema()
task = measAlg.SourceDetectionTask(schema, config=config)
table = afwTable.SourceTable.make(schema)
results = task.makeSourceCatalog(table, exp)
ds9.mtv(image, frame=4)

# Merge them together
print "# Before merge, nSrc =", len(results.sources)
fpSet = results.fpSets.positive
fpSet.merge(results.fpSets.negative, 0, 0, False)
sources = afwTable.SourceCatalog(table)
fpSet.makeSources(sources)
print "# After merge, nSrc =", len(sources)
s = sources[0]
print "# And nPeaks =", len(s.getFootprint().getPeaks())

# Measure dipole at known location
schema  = afwTable.SourceTable.makeMinimalSchema()
msb     = measAlg.MeasureSourcesBuilder()\
            .addAlgorithm(ipDiffim.NaiveDipoleCentroidControl())\
            .addAlgorithm(ipDiffim.NaiveDipoleFluxControl())\
            .addAlgorithm(ipDiffim.PsfDipoleFluxControl())
ms      = msb.build(schema)
table   = afwTable.SourceTable.make(schema)
source  = table.makeRecord()

source.setFootprint(s.getFootprint())
ms.apply(source, exp, afwGeom.Point2D(xc, yc))

for key in schema.getNames():
    print key, source.get(key)


dpDeblender = diffimTools.DipoleDeblender()
deblendSource = dpDeblender(source, exp)

fp     = deblendSource.getFootprint()
peaks  = fp.getPeaks()
speaks = [(p.getPeakValue(), p) for p in peaks]
speaks.sort() 
dpeaks = [speaks[0][1], speaks[-1][1]]
if True:
    negCenter = afwGeom.Point2D(dpeaks[0].getFx(), dpeaks[0].getFy())
    posCenter = afwGeom.Point2D(dpeaks[1].getFx(), dpeaks[1].getFy())
else:
    # Force the known center
    negCenter = afwGeom.Point2D(xn+psfw//2, yn+psfw//2)
    posCenter = afwGeom.Point2D(xp+psfw//2, yp+psfw//2)
print "PEAKS", peaks[0].getFx(), dpeaks[0].getFy(), dpeaks[1].getFx(), dpeaks[1].getFy()
print "REALLY AT", xn+psfw//2, yn+psfw//2, xp+psfw//2, yp+psfw//2

psf = exp.getPsf()
negPsf = psf.computeImage(negCenter, True).convertF()
posPsf = psf.computeImage(posCenter, True).convertF()
montage = afwImage.ImageF(fp.getBBox())
negMont = afwImage.ImageF(fp.getBBox())
posMont = afwImage.ImageF(fp.getBBox())
sx, sy = negPsf.getDimensions()
cx, cy = sx//2, sy//2

# The center of the Psf should be at negCenter, posCenter
negBBox = negPsf.getBBox(afwImage.PARENT)
posBBox = posPsf.getBBox(afwImage.PARENT)
montBBox = montage.getBBox(afwImage.PARENT)

# Portion of the negative Psf that overlaps the montage
negXmin = negBBox.getMinX() if (negBBox.getMinX() > montBBox.getMinX()) else montBBox.getMinX()
negYmin = negBBox.getMinY() if (negBBox.getMinY() > montBBox.getMinY()) else montBBox.getMinY()
negXmax = negBBox.getMaxX() if (negBBox.getMaxX() < montBBox.getMaxX()) else montBBox.getMaxX()
negYmax = negBBox.getMaxY() if (negBBox.getMaxY() < montBBox.getMaxY()) else montBBox.getMaxY()
negOverlapBBox = afwGeom.Box2I(afwGeom.Point2I(negXmin, negYmin), afwGeom.Point2I(negXmax, negYmax))

# Portion of the positivePsf that overlaps the montage
posXmin = posBBox.getMinX() if (posBBox.getMinX() > montBBox.getMinX()) else montBBox.getMinX()
posYmin = posBBox.getMinY() if (posBBox.getMinY() > montBBox.getMinY()) else montBBox.getMinY()
posXmax = posBBox.getMaxX() if (posBBox.getMaxX() < montBBox.getMaxX()) else montBBox.getMaxX()
posYmax = posBBox.getMaxY() if (posBBox.getMaxY() < montBBox.getMaxY()) else montBBox.getMaxY()
posOverlapBBox = afwGeom.Box2I(afwGeom.Point2I(posXmin, posYmin), afwGeom.Point2I(posXmax, posYmax))

negSubim     = type(negPsf)(negPsf, negOverlapBBox, afwImage.PARENT)
montSubim    = type(montage)(montage, negOverlapBBox, afwImage.PARENT)
negMontSubim = type(negMont)(negMont, negOverlapBBox, afwImage.PARENT)
montSubim += negSubim
negMontSubim += negSubim
ds9.mtv(montage, frame=5)

posSubim     = type(posPsf)(posPsf, posOverlapBBox, afwImage.PARENT)
montSubim    = type(montage)(montage, posOverlapBBox, afwImage.PARENT)
posMontSubim = type(posMont)(posMont, posOverlapBBox, afwImage.PARENT)
montSubim += posSubim
posMontSubim += posSubim

data = afwImage.ImageF(exp.getMaskedImage().getImage(), fp.getBBox())
ds9.mtv(montage, frame=6, title="Unfitted model")
ds9.mtv(data, frame=7, title="Data")
#ds9.mtv(negMont, frame=8)
#ds9.mtv(posMont, frame=9)

posPsfSum = np.sum(posPsf.getArray())
negPsfSum = np.sum(negPsf.getArray())

M = np.array((np.ravel(negMont.getArray()), np.ravel(posMont.getArray()))).T.astype(np.float64)
B = np.array((np.ravel(data.getArray()))).astype(np.float64)
matrixNorm = 1. / np.sqrt(np.median(var.getArray()))
M *= matrixNorm
B *= matrixNorm

# SCIPY
fneg, fpos = np.linalg.lstsq(M, B)[0]
print "SCIPY", fneg, fpos, fneg*negPsfSum, fpos*posPsfSum

lsq = afwMath.LeastSquares.fromDesignMatrix(M, B, afwMath.LeastSquares.DIRECT_SVD)
fneg, fpos = lsq.getSolution()
cov        = lsq.getCovariance()
print "AFW", fneg, fpos, fneg*negPsfSum, fpos*posPsfSum, "ERR:", np.sqrt(cov[0][0]), np.sqrt(cov[1][1])

montage      = afwImage.ImageF(fp.getBBox())
negSubim     = type(negPsf)(negPsf, negOverlapBBox, afwImage.PARENT, True)
negSubim    *= float(fneg)
posSubim     = type(posPsf)(posPsf, posOverlapBBox, afwImage.PARENT, True)
posSubim    *= float(fpos)
montSubim    = type(montage)(montage, negOverlapBBox, afwImage.PARENT)
montSubim   += negSubim
montSubim    = type(montage)(montage, posOverlapBBox, afwImage.PARENT)
montSubim   += posSubim

ds9.mtv(montage, frame=8, title="Fitted model")
montage   -= data
ds9.mtv(montage, frame=9, title="Residuals")
montage   *= montage
montage   /= afwImage.ImageF(exp.getMaskedImage().getVariance(), fp.getBBox())
chi2       = np.sum(montage.getArray())
npts       = montage.getArray().shape[0] * montage.getArray().shape[1]
print "MY CHI", chi2 / (npts - 2)

print "AFW", source.get("flux.dipole.psf.neg"), source.get("flux.dipole.psf.pos"), source.get("flux.dipole.psf.chi2dof")
print "AFW", source.get("flux.dipole.psf.neg.centroid"), source.get("flux.dipole.psf.pos.centroid")


####

negPsf = psf.computeImage(source.get("flux.dipole.psf.neg.centroid"), True).convertF()
posPsf = psf.computeImage(source.get("flux.dipole.psf.pos.centroid"), True).convertF()
negPsf /= float(np.sum(negPsf.getArray()))
posPsf /= float(np.sum(posPsf.getArray()))

fpos = source.get("flux.dipole.psf.pos")
fneg = source.get("flux.dipole.psf.neg")
negBBox = negPsf.getBBox(afwImage.PARENT)
posBBox = posPsf.getBBox(afwImage.PARENT)

# Portion of the negative Psf that overlaps the montage
negXmin = negBBox.getMinX() if (negBBox.getMinX() > montBBox.getMinX()) else montBBox.getMinX()
negYmin = negBBox.getMinY() if (negBBox.getMinY() > montBBox.getMinY()) else montBBox.getMinY()
negXmax = negBBox.getMaxX() if (negBBox.getMaxX() < montBBox.getMaxX()) else montBBox.getMaxX()
negYmax = negBBox.getMaxY() if (negBBox.getMaxY() < montBBox.getMaxY()) else montBBox.getMaxY()
negOverlapBBox = afwGeom.Box2I(afwGeom.Point2I(negXmin, negYmin), afwGeom.Point2I(negXmax, negYmax))

# Portion of the positivePsf that overlaps the montage
posXmin = posBBox.getMinX() if (posBBox.getMinX() > montBBox.getMinX()) else montBBox.getMinX()
posYmin = posBBox.getMinY() if (posBBox.getMinY() > montBBox.getMinY()) else montBBox.getMinY()
posXmax = posBBox.getMaxX() if (posBBox.getMaxX() < montBBox.getMaxX()) else montBBox.getMaxX()
posYmax = posBBox.getMaxY() if (posBBox.getMaxY() < montBBox.getMaxY()) else montBBox.getMaxY()
posOverlapBBox = afwGeom.Box2I(afwGeom.Point2I(posXmin, posYmin), afwGeom.Point2I(posXmax, posYmax))

montage      = afwImage.ImageF(fp.getBBox())
negSubim     = type(negPsf)(negPsf, negOverlapBBox, afwImage.PARENT, True)
negSubim    *= float(fneg)
posSubim     = type(posPsf)(posPsf, posOverlapBBox, afwImage.PARENT, True)
posSubim    *= float(fpos)
montSubim    = type(montage)(montage, negOverlapBBox, afwImage.PARENT)
montSubim   += negSubim
montSubim    = type(montage)(montage, posOverlapBBox, afwImage.PARENT)
montSubim   += posSubim

montage   -= data
ds9.mtv(montage, frame=10, title="New Residuals")
montage   *= montage
montage   /= afwImage.ImageF(exp.getMaskedImage().getVariance(), fp.getBBox())
chi2       = np.sum(montage.getArray())
npts       = montage.getArray().shape[0] * montage.getArray().shape[1]
print "MY CHI", chi2 / (npts - 2)


#negBBox.shift(psfShift)
#posBBox.shift(psfShift)

# Place the center of the Psf at the fitted coords
#negBBox.shift(afwGeom.Extent2I(afwGeom.Point2I(negCenter)))
#posBBox.shift(afwGeom.Extent2I(afwGeom.Point2I(posCenter)))
    

#### Work on C++ dipole measurement here
import pdb; pdb.set_trace()






