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

#include "lsst/afw/image/ImagePca.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/SpatialCell.h"
#include "lsst/ip/diffim/KernelPca.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace ip {
namespace diffim {
namespace detail {

namespace {

/**
 * Wrap KernelPca for one pixel type
 *
 * @tparam PixelT  Image pixel type, e.g. `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Class name suffix associated with PixeT, e.g. "F" for `float`
 */
template <typename PixelT>
void declareKernelPca(lsst::cpputils::python::WrapperCollection &wrappers, std::string const& suffix) {
    using ImageT = afw::image::Image<PixelT>;
    using PyClass = py::class_<KernelPca<ImageT>, afw::image::ImagePca<ImageT>>;

    std::string name = "KernelPca" + suffix;
    wrappers.wrapType(PyClass(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {

        cls.def(py::init<bool>(), "constantWeight"_a = true);

        cls.def("analyze", &KernelPca<ImageT>::analyze);
    });
}

/**
 * Wrap KernelPcaVisitor for one pixel type
 *
 * @tparam PixelT  Image pixel type, e.g. `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Class name suffix associated with PixeT, e.g. "F" for `float`
 */
template <typename PixelT>
void declareKernelPcaVisitor(lsst::cpputils::python::WrapperCollection &wrappers, std::string const& suffix) {
    using PyClass = py::class_<KernelPcaVisitor<PixelT>, afw::math::CandidateVisitor>;

    std::string name = "KernelPcaVisitor" + suffix;
    // note that KernelPcaVisitor<PixelT>::ImageT
    // is always the same (pixels of type lsst::afw::math::Kernel::Pixel, not PixelT)
    using KernelImageT = typename KernelPcaVisitor<PixelT>::ImageT;

    wrappers.wrapType(PyClass(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<std::shared_ptr<KernelPca<KernelImageT>>>(), "imagePca"_a);

        cls.def("getEigenKernels", &KernelPcaVisitor<PixelT>::getEigenKernels);
        cls.def("processCandidate", &KernelPcaVisitor<PixelT>::processCandidate, "candidate"_a);
        cls.def("subtractMean", &KernelPcaVisitor<PixelT>::subtractMean);
        cls.def("returnMean", &KernelPcaVisitor<PixelT>::returnMean);

        mod.def("makeKernelPcaVisitor", &makeKernelPcaVisitor<PixelT>, "imagePca"_a);
    });
}

}  // namespace lsst::ip::diffim::detail::<anonymous>

void wrapKernelPca(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareKernelPca<afw::math::Kernel::Pixel>(wrappers, "D");
    declareKernelPcaVisitor<float>(wrappers, "F");
}

}  // detail
}  // diffim
}  // ip
}  // lsst::ip::diffim::detail
