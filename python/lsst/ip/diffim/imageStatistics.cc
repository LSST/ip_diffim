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

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/ip/diffim/ImageStatistics.h"
#include "lsst/daf/base/PropertySet.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace ip {
namespace diffim {

namespace {

/**
 * Wrap ImageStatistics for one image plane pixel type
 *
 * @tparam PixelT  Image plane pixel type, e.g. `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Class name suffix associated with PixeT, e.g. "F" for `float`
 */
template <typename PixelT>
void declareImageStatistics(lsst::cpputils::python::WrapperCollection &wrappers, std::string const &suffix) {
    using PyImageStatistics =  py::class_<ImageStatistics<PixelT>>;

    std::string name = "ImageStatistics" + suffix;
    wrappers.wrapType(PyImageStatistics(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<daf::base::PropertySet const &>(), "ps"_a);

        cls.def("reset", &ImageStatistics<PixelT>::reset);
        cls.def("apply", (void (ImageStatistics<PixelT>::*)(afw::image::MaskedImage<PixelT> const &)) &
                        ImageStatistics<PixelT>::apply,
                "image"_a);
        cls.def("apply", (void (ImageStatistics<PixelT>::*)(afw::image::MaskedImage<PixelT> const &, int)) &
                        ImageStatistics<PixelT>::apply,
                "image"_a, "core"_a);
        cls.def("setBpMask", &ImageStatistics<PixelT>::setBpMask, "bpMask"_a);
        cls.def("getBpMask", &ImageStatistics<PixelT>::getBpMask);
        cls.def("getMean", &ImageStatistics<PixelT>::getMean);
        cls.def("getVariance", &ImageStatistics<PixelT>::getVariance);
        cls.def("getRms", &ImageStatistics<PixelT>::getRms);
        cls.def("getNpix", &ImageStatistics<PixelT>::getNpix);
        cls.def("evaluateQuality", &ImageStatistics<PixelT>::evaluateQuality, "ps"_a);
    });
}

}  // namespace lsst::ip::diffim::<anonymous>

void wrapImageStatistics(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareImageStatistics<int>(wrappers, "I");
    declareImageStatistics<float>(wrappers, "F");
    declareImageStatistics<double>(wrappers, "D");
}

}  // diffim
}  // ip
}  // lsst
