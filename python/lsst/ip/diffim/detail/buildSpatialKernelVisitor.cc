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
#include "pybind11/stl.h"

#include <memory>
#include <string>

#include <Eigen/Core>
#include "ndarray/pybind11.h"

#include "lsst/afw/math.h"
#include "lsst/ip/diffim/BuildSpatialKernelVisitor.h"
#include "lsst/daf/base/PropertySet.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace ip {
namespace diffim {
namespace detail {

namespace {

/**
 * Wrap BuildSpatialKernelVisitor and the factory function for one pixel type
 *
 * @tparam PixelT  Pixel type, e.g. `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Class name suffix associated with PixeT, e.g. "F" for `float`
 */
template <typename PixelT>
void declareBuildSpatialKernelVisitor(lsst::cpputils::python::WrapperCollection &wrappers, std::string const& suffix) {
    std::string name = "BuildSpatialKernelVisitor" + suffix;
    using PyClass = py::class_<BuildSpatialKernelVisitor<PixelT>, afw::math::CandidateVisitor>;
    wrappers.wrapType(PyClass(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<afw::math::KernelList, geom::Box2I const &, daf::base::PropertySet const &>(), "basisList"_a,
                "regionBBox"_a, "ps"_a);

        cls.def("getNCandidates", &BuildSpatialKernelVisitor<PixelT>::getNCandidates);
        cls.def("processCandidate", &BuildSpatialKernelVisitor<PixelT>::processCandidate, "candidate"_a);
        cls.def("solveLinearEquation", &BuildSpatialKernelVisitor<PixelT>::solveLinearEquation);
        cls.def("getKernelSolution", &BuildSpatialKernelVisitor<PixelT>::getKernelSolution);
        cls.def("getSolutionPair", &BuildSpatialKernelVisitor<PixelT>::getSolutionPair);

        mod.def("makeBuildSpatialKernelVisitor", &makeBuildSpatialKernelVisitor<PixelT>, "basisList"_a,
                "regionBBox"_a, "ps"_a);
    });
}

}  // namespace lsst::ip::diffim::detail::<anonymous>

void wrapBuildSpatialKernelVisitor(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareBuildSpatialKernelVisitor<float>(wrappers, "F");
}

}  // detail
}  // diffim
}  // ip
}  // lsst
