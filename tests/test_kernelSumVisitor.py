import unittest


import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.ip.diffim as ipDiffim
import lsst.log.utils as logUtils
import lsst.pex.config as pexConfig

logUtils.traceSetAt("ip.diffim", 2)


class DiffimTestCases(lsst.utils.tests.TestCase):

    def setUp(self):
        self.config = ipDiffim.ImagePsfMatchTask.ConfigClass()
        self.config.kernel.name = "DF"
        self.subconfig = self.config.kernel.active

        self.policy = pexConfig.makePolicy(self.subconfig)
        self.kList = ipDiffim.makeKernelBasisList(self.subconfig)

    def makeCandidate(self, kSum, x, y, size=51):
        mi1 = afwImage.MaskedImageF(geom.Extent2I(size, size))
        mi1.getVariance().set(1.0)  # avoid NaNs
        mi1[size//2, size//2, afwImage.LOCAL] = (1, 0x0, 1)
        mi2 = afwImage.MaskedImageF(geom.Extent2I(size, size))
        mi2.getVariance().set(1.0)  # avoid NaNs
        mi2[size//2, size//2, afwImage.LOCAL] = (kSum, 0x0, kSum)
        kc = ipDiffim.makeKernelCandidate(x, y, mi1, mi2, self.policy)
        return kc

    def tearDown(self):
        del self.policy
        del self.kList

    def testAggregate(self, kSums=[1., 1., 1., 1., 2., 3., 4.]):
        ksv = ipDiffim.KernelSumVisitorF(self.policy)
        ksv.setMode(ipDiffim.KernelSumVisitorF.AGGREGATE)

        # should fail, kernel not initialized
        kc = self.makeCandidate(1, 0.0, 0.0)
        try:
            ksv.processCandidate(kc)
        except Exception:
            pass
        else:
            self.fail()

        for kSum in kSums:
            kc = self.makeCandidate(kSum, 0., 0.)
            kc.build(self.kList)
            self.assertAlmostEqual(kSum, kc.getKsum(ipDiffim.KernelCandidateF.RECENT))
            ksv.processCandidate(kc)

        for method in (ksv.getNRejected,
                       ksv.getkSumMean,
                       ksv.getkSumStd,
                       ksv.getdkSumMax,
                       ksv.getkSumNpts):
            self.assertEqual(method(), 0.0)

        ksv.processKsumDistribution()

        self.assertEqual(ksv.getNRejected(), 0)
        self.assertAlmostEqual(ksv.getkSumMean(),
                               afwMath.makeStatistics(kSums, afwMath.MEANCLIP).getValue(afwMath.MEANCLIP))
        self.assertAlmostEqual(ksv.getkSumStd(),
                               afwMath.makeStatistics(kSums, afwMath.STDEVCLIP).getValue(afwMath.STDEVCLIP))
        self.assertEqual(ksv.getdkSumMax(),
                         self.policy.get("maxKsumSigma") * ksv.getkSumStd())
        self.assertEqual(ksv.getkSumNpts(), len(kSums))

    def testReject(self):
        self.doReject(clipping=False)
        self.doReject(clipping=True)

    def doReject(self, clipping, kSums=[1., 1., 1., 1., 2., 3., 4., 50.]):
        self.policy.set("kernelSumClipping", clipping)
        ksv = ipDiffim.KernelSumVisitorF(self.policy)
        ksv.setMode(ipDiffim.KernelSumVisitorF.AGGREGATE)
        kcList = []

        for kSum in kSums:
            kc = self.makeCandidate(kSum, 0., 0.)
            kc.build(self.kList)
            kc.setStatus(afwMath.SpatialCellCandidate.GOOD)
            self.assertAlmostEqual(kSum, kc.getKsum(ipDiffim.KernelCandidateF.RECENT))
            ksv.processCandidate(kc)
            kcList.append(kc)

        ksv.processKsumDistribution()

        ksv.setMode(ipDiffim.KernelSumVisitorF.REJECT)
        for kc in kcList:
            ksv.processCandidate(kc)
            if clipping and kc == kcList[-1]:
                self.assertEqual(kc.getStatus(), afwMath.SpatialCellCandidate.BAD)
            else:
                self.assertEqual(kc.getStatus(), afwMath.SpatialCellCandidate.GOOD)

        if clipping:
            self.assertEqual(ksv.getNRejected(), 1)
        else:
            self.assertEqual(ksv.getNRejected(), 0)

    def testVisit(self, nCell=3):
        ksv = ipDiffim.makeKernelSumVisitor(self.policy)

        sizeCellX = self.policy.get("sizeCellX")
        sizeCellY = self.policy.get("sizeCellY")

        kernelCellSet = afwMath.SpatialCellSet(geom.Box2I(geom.Point2I(0, 0),
                                                          geom.Extent2I(sizeCellX * nCell,
                                                                        sizeCellY * nCell)),
                                               sizeCellX,
                                               sizeCellY)

        for candX in range(nCell):
            for candY in range(nCell):
                if candX == nCell // 2 and candY == nCell // 2:
                    kc = self.makeCandidate(100.0,
                                            candX * sizeCellX + sizeCellX // 2,
                                            candY * sizeCellY + sizeCellY // 2)
                else:
                    kc = self.makeCandidate(1.0,
                                            candX * sizeCellX + sizeCellX // 2,
                                            candY * sizeCellY + sizeCellY // 2)
                kc.build(self.kList)
                kernelCellSet.insertCandidate(kc)

        ksv.setMode(ipDiffim.KernelSumVisitorF.AGGREGATE)
        kernelCellSet.visitCandidates(ksv, 1)
        ksv.processKsumDistribution()
        ksv.setMode(ipDiffim.KernelSumVisitorF.REJECT)
        kernelCellSet.visitCandidates(ksv, 1)

        self.assertEqual(ksv.getNRejected(), 1)

#####


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
