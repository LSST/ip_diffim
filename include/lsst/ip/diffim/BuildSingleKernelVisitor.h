// -*- lsst-c++ -*-
/**
 * @file BuildSingleKernelVisitor.h
 *
 * @brief Declaration of BuildSingleKernelVisitor
 *
 * @author Andrew Becker, University of Washington
 *
 * @ingroup ip_diffim
 */

#ifndef LSST_IP_DIFFIM_BUILDSINGLEKERNELVISITOR_H
#define LSST_IP_DIFFIM_BUILDSINGLEKERNELVISITOR_H

#include <memory>

#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

#include "lsst/daf/base/PropertySet.h"

#include "lsst/ip/diffim/ImageStatistics.h"

namespace lsst {
namespace ip {
namespace diffim {
namespace detail {

    template<typename PixelT>
    class BuildSingleKernelVisitor : public lsst::afw::math::CandidateVisitor {
        typedef lsst::afw::image::MaskedImage<PixelT> MaskedImageT;
    public:
        typedef std::shared_ptr<BuildSingleKernelVisitor<PixelT> > Ptr;

        BuildSingleKernelVisitor(
            lsst::afw::math::KernelList const& basisList,
            lsst::daf::base::PropertySet const& ps
            );
        BuildSingleKernelVisitor(
            lsst::afw::math::KernelList const& basisList,
            lsst::daf::base::PropertySet const& ps,
            Eigen::MatrixXd const& hMat
            );
        virtual ~BuildSingleKernelVisitor() {};

        /*
           Don't reprocess candidate if its already been build.  The use
           case for this functionality is : when iterating over all Cells
           and rejecting bad Kernels, we need to re-visit *all* Cells to
           build the next candidate in the list.  Without this flag we would
           unncessarily re-build all the good Kernels.
        */
        void setSkipBuilt(bool skip)      {_skipBuilt = skip;}

        int getNRejected()    {return _nRejected;}
        int getNProcessed()   {return _nProcessed;}
        void reset()          {_nRejected = 0; _nProcessed = 0;}

        std::shared_ptr<lsst::daf::base::PropertySet> getPropertySet() { return _ps; }
        bool getSkipBuilt() { return _skipBuilt; }
        bool getUseRegularization() { return _useRegularization; }
        bool getUseCoreStats() { return _useCoreStats; }
        int getCoreRadius() { return _coreRadius; }

        void processCandidate(lsst::afw::math::SpatialCellCandidate *candidate);

    private:
        lsst::afw::math::KernelList const _basisList; ///< Basis set
        lsst::daf::base::PropertySet::Ptr _ps; ///< PS controlling behavior
        Eigen::MatrixXd const _hMat;          ///< Regularization matrix
        ImageStatistics<PixelT> _imstats;     ///< To calculate statistics of difference image
        bool _skipBuilt;                      ///< Skip over built candidates during processCandidate()
        int _nRejected;                       ///< Number of candidates rejected during processCandidate()
        int _nProcessed;                      ///< Number of candidates processed during processCandidate()
        bool _useRegularization;              ///< Regularize if delta function basis

        bool _useCoreStats;                   ///< Extracted from _ps
        int _coreRadius;                      ///< Extracted from _ps
    };

    template<typename PixelT>
    std::shared_ptr<BuildSingleKernelVisitor<PixelT> >
    makeBuildSingleKernelVisitor(
        lsst::afw::math::KernelList const& basisList,
        lsst::daf::base::PropertySet const& ps
        ) {

        return std::shared_ptr<BuildSingleKernelVisitor<PixelT>>(
            new BuildSingleKernelVisitor<PixelT>(basisList, ps)
            );
    }

    template<typename PixelT>
    std::shared_ptr<BuildSingleKernelVisitor<PixelT> >
    makeBuildSingleKernelVisitor(
        lsst::afw::math::KernelList const& basisList,
        lsst::daf::base::PropertySet const& ps,
        Eigen::MatrixXd const & hMat
        ) {

        return std::shared_ptr<BuildSingleKernelVisitor<PixelT>>(
            new BuildSingleKernelVisitor<PixelT>(basisList, ps, hMat)
            );
    }

}}}} // end of namespace lsst::ip::diffim::detail

#endif
