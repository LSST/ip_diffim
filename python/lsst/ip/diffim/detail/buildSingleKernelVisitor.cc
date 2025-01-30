/*
 * LSST Data Management System
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 * See the COPYRIGHT file
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */
#include "pybind11/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"

#include <memory>
#include <string>

#include <Eigen/Core>
#include "ndarray/pybind11.h"

#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/SpatialCell.h"
#include "lsst/ip/diffim/BuildSingleKernelVisitor.h"
#include "lsst/daf/base/PropertySet.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace ip {
namespace diffim {
namespace detail {

namespace {

/**
 * Wrap BuildSingleKernelVisitor and the factory function for one pixel type
 *
 * @tparam PixelT  Pixel type, e.g. `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Class name suffix associated with PixeT, e.g. "F" for `float`
 */
template <typename PixelT>
void declareBuildSingleKernelVisitor(lsst::cpputils::python::WrapperCollection &wrappers, std::string const& suffix) {
    using PyClass = py::class_<BuildSingleKernelVisitor<PixelT>, std::shared_ptr<BuildSingleKernelVisitor<PixelT>>,
               afw::math::CandidateVisitor>;
    std::string name = "BuildSingleKernelVisitor" + suffix;

    wrappers.wrapType(PyClass(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<afw::math::KernelList, daf::base::PropertySet const &>(), "basisList"_a, "ps"_a);
        cls.def(py::init<afw::math::KernelList, daf::base::PropertySet const &, Eigen::MatrixXd const &>(),
                "basisList"_a,
                "ps"_a, "hMat"_a);

        cls.def("setSkipBuilt", &BuildSingleKernelVisitor<PixelT>::setSkipBuilt, "skip"_a);
        cls.def("getNRejected", &BuildSingleKernelVisitor<PixelT>::getNRejected);
        cls.def("getNProcessed", &BuildSingleKernelVisitor<PixelT>::getNProcessed);
        cls.def("reset", &BuildSingleKernelVisitor<PixelT>::reset);
        cls.def("processCandidate", &BuildSingleKernelVisitor<PixelT>::processCandidate, "candidate"_a);
        cls.def_property_readonly("nRejected", &BuildSingleKernelVisitor<PixelT>::getNRejected);
        cls.def_property_readonly("nProcessed", &BuildSingleKernelVisitor<PixelT>::getNProcessed);
        cls.def_property_readonly("useRegularization", &BuildSingleKernelVisitor<PixelT>::getUseRegularization);
        cls.def_property_readonly("skipBuilt", &BuildSingleKernelVisitor<PixelT>::getSkipBuilt);
        cls.def_property_readonly("useCoreStats", &BuildSingleKernelVisitor<PixelT>::getUseCoreStats);
        cls.def_property_readonly("coreRadius", &BuildSingleKernelVisitor<PixelT>::getCoreRadius);
        cls.def_property_readonly("propertySet", &BuildSingleKernelVisitor<PixelT>::getPropertySet);

        mod.def("makeBuildSingleKernelVisitor",
                (std::shared_ptr<BuildSingleKernelVisitor<PixelT>>(*)(afw::math::KernelList const &,
                                                                      daf::base::PropertySet const &)) &
                        makeBuildSingleKernelVisitor<PixelT>,
                "basisList"_a, "ps"_a);
        mod.def("makeBuildSingleKernelVisitor",
                (std::shared_ptr<BuildSingleKernelVisitor<PixelT>>(*)(
                        afw::math::KernelList const &, daf::base::PropertySet const &, Eigen::MatrixXd const &)) &
                        makeBuildSingleKernelVisitor<PixelT>,
                "basisList"_a, "ps"_a, "hMat"_a);
    });
}

}  // namespace lsst::ip::diffim::detail::<anonymous>

void wrapBuildSingleKernelVisitor(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareBuildSingleKernelVisitor<float>(wrappers, "F");
}

}  // detail
}  // diffim
}  // ip
}  // lsst
