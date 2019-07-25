// -*- lsst-c++ -*-
/**
 * @file BuildSingleKernelVisitor.h
 *
 * @brief Implementation of BuildSingleKernelVisitor
 *
 * @author Andrew Becker, University of Washington
 *
 * @ingroup ip_diffim
 */

#include <memory>
#include "Eigen/Core"

#include "lsst/afw/math.h"
#include "lsst/afw/image.h"
#include "lsst/log/Log.h"
#include "lsst/pex/exceptions/Runtime.h"
#include "lsst/pex/policy/Policy.h"

#include "lsst/ip/diffim/ImageSubtract.h"
#include "lsst/ip/diffim/KernelCandidate.h"
#include "lsst/ip/diffim/BuildSingleKernelVisitor.h"

#define DEBUG_MATRIX 0

namespace afwMath        = lsst::afw::math;
namespace afwImage       = lsst::afw::image;
namespace pexPolicy      = lsst::pex::policy;
namespace pexExcept      = lsst::pex::exceptions;
namespace ipDiffim       = lsst::ip::diffim;

namespace lsst {
namespace ip {
namespace diffim {
namespace detail {

    /**
     * @class BuildSingleKernelVisitor
     * @ingroup ip_diffim
     *
     * @brief Builds the convolution kernel for a given candidate
     *
     * @code
        std::shared_ptr<Policy> policy(new Policy);
        policy->set("constantVarianceWeighting", false);
        policy->set("iterateSingleKernel", false);
        policy->set("singleKernelClipping", true);
        policy->set("candidateResidualMeanMax", 0.25);
        policy->set("candidateResidualStdMax", 1.25);

        detail::BuildSingleKernelVisitor<PixelT> singleKernelFitter(*policy);
        int nRejected = -1;
        while (nRejected != 0) {
            singleKernelFitter.reset();
            kernelCells.visitCandidates(&singleKernelFitter, nStarPerCell);
            nRejected = singleKernelFitter.getNRejected();
        }
     * @endcode
     *
     * @note Visits each current candidate in a afwMath::SpatialCellSet, and
     * builds its kernel using its build() method.  We don't build the kernel
     * for *every* candidate since this is computationally expensive, only when
     * its the current candidate in the cell.  During the course of building the
     * kernel, it also assesses the quality of the difference image.  If it is
     * determined to be bad (based on the Policy paramters) the candidate is
     * flagged as afwMath::SpatialCellCandidate::BAD; otherwise its marked as
     * afwMath::SpatialCellCandidate::GOOD.  Keeps a running sample of all the
     * new candidates it visited that turned out to be bad.
     *
     * @note Because this visitor does not have access to the next candidate in the
     * cell, it must be called iteratively until no candidates are rejected.  This
     * ensures that the current candidate of every cell has an initialized Kernel.
     * This also requires that this class re-Visit all the cells after any other
     * Visitors with the ability to mark something as BAD.
     *
     * @note Because we are frequently re-Visiting entirely GOOD candidates during
     * these iterations, the option of _skipBuilt=true will enable the user to *not*
     * rebuilt the kernel on every visit.
     *
     * @note For the particular use case of creating a Pca basis from the raw
     * kernels, we want to re-Visit each candidate and re-fit the kernel using
     * this Pca basis.  This requires the user to setSkipBuilt(false) so that
     * the candidate is reprocessed with this new basis.
     *
     */
    template<typename PixelT>
    BuildSingleKernelVisitor<PixelT>::BuildSingleKernelVisitor(
        lsst::afw::math::KernelList const& basisList,   ///< List of basis kernels
            ///< for resulting LinearCombinationKernel
        lsst::pex::policy::Policy const& policy  ///< Policy file directing behavior
        ) :
        afwMath::CandidateVisitor(),
        _basisList(basisList),
        _policy(policy),
        _hMat(),
        _imstats(ImageStatistics<PixelT>(_policy)),
        _skipBuilt(true),
        _nRejected(0),
        _nProcessed(0),
        _useRegularization(false),
        _useCoreStats(_policy.getBool("useCoreStats")),
        _coreRadius(_policy.getInt("candidateCoreRadius"))
    {};

    template<typename PixelT>
    BuildSingleKernelVisitor<PixelT>::BuildSingleKernelVisitor(
        lsst::afw::math::KernelList const& basisList,   ///< List of basis kernels
            ///< for resulting LinearCombinationKernel
        lsst::pex::policy::Policy const& policy,  ///< Policy file directing behavior
        Eigen::MatrixXd const& hMat   ///< Regularization matrix
        ) :
        afwMath::CandidateVisitor(),
        _basisList(basisList),
        _policy(policy),
        _hMat(hMat),
        _imstats(ImageStatistics<PixelT>(_policy)),
        _skipBuilt(true),
        _nRejected(0),
        _nProcessed(0),
        _useRegularization(true),
        _useCoreStats(_policy.getBool("useCoreStats")),
        _coreRadius(_policy.getInt("candidateCoreRadius"))
    {};


    template<typename PixelT>
    void BuildSingleKernelVisitor<PixelT>::processCandidate(
        lsst::afw::math::SpatialCellCandidate *candidate
        ) {

        ipDiffim::KernelCandidate<PixelT> *kCandidate =
            dynamic_cast<ipDiffim::KernelCandidate<PixelT> *>(candidate);
        if (kCandidate == NULL) {
            LOGL_DEBUG("TRACE2.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                       "Failed to cast SpatialCellCandidate to KernelCandidate %d",
                       kCandidate->getId());
            throw LSST_EXCEPT(pexExcept::LogicError,
                              "Failed to cast SpatialCellCandidate to KernelCandidate");
        }

        if (_skipBuilt and kCandidate->isInitialized()) {
            return;
        }

        LOGL_DEBUG("TRACE1.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                   "Processing candidate %d", kCandidate->getId());
        LOGL_DEBUG("TRACE4.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                   "X = %.2f Y = %.2f",
                   kCandidate->getXCenter(),
                   kCandidate->getYCenter());

        /* Build its kernel here */
        try {
            if (_useRegularization)
                kCandidate->build(_basisList, _hMat);
            else
                kCandidate->build(_basisList);

        } catch (pexExcept::Exception &e) {
            kCandidate->setStatus(afwMath::SpatialCellCandidate::BAD);
            LOGL_DEBUG("TRACE3.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                       "Unable to process candidate %d; exception caught (%s)",
                       kCandidate->getId(),
                       e.what());
            _nRejected += 1;
            return;
        }

        if (kCandidate->getStatus() == afwMath::SpatialCellCandidate::BAD) {
            LOGL_DEBUG("TRACE3.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                       "Candidate %d Returned BAD upon build, exiting",
                       kCandidate->getId());
            _nRejected += 1;
            return;
        }


         /*
         * Make diffim and set chi2 from result.  Note that you need to use the
         * most recent kernel
         */
        MaskedImageT diffim = kCandidate->getDifferenceImage(ipDiffim::KernelCandidate<PixelT>::RECENT);
        try {
            if (_useCoreStats)
                _imstats.apply(diffim, _coreRadius);
            else
                _imstats.apply(diffim);
        } catch (pexExcept::Exception& e) {
            LOGL_DEBUG("TRACE2.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                       "Unable to calculate imstats for Candidate %d", kCandidate->getId());
            kCandidate->setStatus(afwMath::SpatialCellCandidate::BAD);
            return;
        }
        _nProcessed += 1;

        kCandidate->setChi2(_imstats.getVariance());

        /* When using a Pca basis, we don't reset the kernel or background,
           so we need to evaluate these locally for the Trace */
        double kSum = kCandidate->getKsum(ipDiffim::KernelCandidate<PixelT>::RECENT);
        double background = kCandidate->getBackground(ipDiffim::KernelCandidate<PixelT>::RECENT);

        LOGL_DEBUG("TRACE4.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                   "Chi2 = %.3f", kCandidate->getChi2());
        LOGL_DEBUG("TRACE4.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                   "Kernel Sum = %.3f", kSum);
        LOGL_DEBUG("TRACE4.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                   "Background = %.3f", background);
        LOGL_DEBUG("TRACE2.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                   "Candidate %d resids = %.3f +/- %.3f sigma (%d pix)",
                   kCandidate->getId(),
                   _imstats.getMean(),
                   _imstats.getRms(),
                   _imstats.getNpix());

        bool meanIsNan = std::isnan(_imstats.getMean());
        bool rmsIsNan  = std::isnan(_imstats.getRms());
        if (meanIsNan || rmsIsNan) {
            kCandidate->setStatus(afwMath::SpatialCellCandidate::BAD);
            LOGL_DEBUG("TRACE3.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                       "Rejecting candidate %d, encountered NaN",
                       kCandidate->getId());
            _nRejected += 1;
            return;
        }

        if (_policy.getBool("singleKernelClipping")) {
            if (fabs(_imstats.getMean()) > _policy.getDouble("candidateResidualMeanMax")) {
                kCandidate->setStatus(afwMath::SpatialCellCandidate::BAD);
                LOGL_DEBUG("TRACE3.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                           "Rejecting candidate %d; bad mean residual : |%.3f| > %.3f",
                           kCandidate->getId(),
                           _imstats.getMean(),
                           _policy.getDouble("candidateResidualMeanMax"));
                _nRejected += 1;
            }
            else if (_imstats.getRms() > _policy.getDouble("candidateResidualStdMax")) {
                kCandidate->setStatus(afwMath::SpatialCellCandidate::BAD);
                LOGL_DEBUG("TRACE3.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                           "Rejecting candidate %d; bad residual rms : %.3f > %.3f",
                           kCandidate->getId(),
                           _imstats.getRms(),
                           _policy.getDouble("candidateResidualStdMax"));
                _nRejected += 1;
            }
            else {
                kCandidate->setStatus(afwMath::SpatialCellCandidate::GOOD);
                LOGL_DEBUG("TRACE3.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                           "Source kernel OK");
            }
        }
        else {
            kCandidate->setStatus(afwMath::SpatialCellCandidate::GOOD);
            LOGL_DEBUG("TRACE5.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                       "Sigma clipping not enabled");
        }

        /* Core resids for debugging */
        if (!(_useCoreStats)) {
            try {
                _imstats.apply(diffim, _coreRadius);
            } catch (pexExcept::Exception& e) {
                LOGL_DEBUG("TRACE2.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                           "Unable to calculate core imstats for Candidate %d",
                           kCandidate->getId());
                kCandidate->setStatus(afwMath::SpatialCellCandidate::BAD);
                return;
            }
            LOGL_DEBUG("TRACE3.ip.diffim.BuildSingleKernelVisitor.processCandidate",
                       "Candidate %d core resids = %.3f +/- %.3f sigma (%d pix)",
                       kCandidate->getId(),
                       _imstats.getMean(),
                       _imstats.getRms(),
                       _imstats.getNpix());
        }

    }

    typedef float PixelT;

    template class BuildSingleKernelVisitor<PixelT>;

    template std::shared_ptr<BuildSingleKernelVisitor<PixelT> >
    makeBuildSingleKernelVisitor<PixelT>(lsst::afw::math::KernelList const&,
                                         lsst::pex::policy::Policy const&);

    template std::shared_ptr<BuildSingleKernelVisitor<PixelT> >
    makeBuildSingleKernelVisitor<PixelT>(lsst::afw::math::KernelList const&,
                                         lsst::pex::policy::Policy const&,
                                         Eigen::MatrixXd const &);

}}}} // end of namespace lsst::ip::diffim::detail
