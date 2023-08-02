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

#include <memory>
#include <string>

#include "lsst/afw/math/SpatialCell.h"
#include "lsst/ip/diffim/KernelSumVisitor.h"
#include "lsst/daf/base/PropertySet.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace ip {
namespace diffim {
namespace detail {

namespace {

/**
 * Wrap KernelSumVisitor and the factory function for one pixel type
 *
 * @tparam PixelT  Pixel type, e.g. `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Class name suffix associated with PixeT, e.g. "F" for `float`
 */
template <typename PixelT>
void declareKernelSumVisitor(lsst::cpputils::python::WrapperCollection &wrappers, std::string const& suffix) {
    using Class = KernelSumVisitor<PixelT>;

    using PyClass = py::class_<Class, afw::math::CandidateVisitor>;
    std::string name = "KernelSumVisitor" + suffix;
    auto clsDef = wrappers.wrapType(PyClass(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<daf::base::PropertySet const &>(), "ps"_a);

        cls.def("setMode", &Class::setMode, "mode"_a);
        cls.def("getNRejected", &Class::getNRejected);
        cls.def("getkSumMean", &Class::getkSumMean);
        cls.def("getkSumStd", &Class::getkSumStd);
        cls.def("getdkSumMax", &Class::getdkSumMax);
        cls.def("getkSumNpts", &Class::getkSumNpts);
        cls.def("resetKernelSum", &Class::resetKernelSum);
        cls.def("processCandidate", &Class::processCandidate, "candidate"_a);
        cls.def("processKsumDistribution", &Class::processKsumDistribution);

        mod.def("makeKernelSumVisitor", &makeKernelSumVisitor<PixelT>, "ps"_a);
    });

    wrappers.wrapType(py::enum_<typename Class::Mode>(clsDef, "Mode"), [](auto &mod, auto &enm) {
        enm.value("AGGREGATE", Class::Mode::AGGREGATE);
        enm.value("REJECT", Class::Mode::REJECT);
        enm.export_values();
    });
}

}  // namespace lsst::ip::diffim::detail::<anonymous>

void wrapKernelSumVisitor(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareKernelSumVisitor<float>(wrappers, "F");
}

}  // detail
}  // diffim
}  // ip
}  // lsst
