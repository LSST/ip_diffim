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

#include "lsst/ip/diffim/KernelCandidateDetection.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace ip {
namespace diffim {

namespace {

/**
 * Wrap `KernelCandidateDetection` for one pixel type
 *
 * @tparam PixelT  Pixel type of image plane of MaskedImage, e.g. `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Class name suffix associated with PixeT, e.g. "F" for `float`
 */
template <typename PixelT>
void declareKernelCandidateDetection(lsst::cpputils::python::WrapperCollection &wrappers, std::string const &suffix) {
    using PyKernelCandidateDetection =
            py::class_<KernelCandidateDetection<PixelT>>;

    std::string name = "KernelCandidateDetection" + suffix;
    wrappers.wrapType(PyKernelCandidateDetection(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<daf::base::PropertySet const &>(), "ps"_a);

        cls.def("apply", &KernelCandidateDetection<PixelT>::apply, "templateMaskedImage"_a,
                "scienceMaskedImage"_a);
        cls.def("growCandidate", &KernelCandidateDetection<PixelT>::growCandidate, "footprint"_a, "fpGrowPix"_a,
                "templateMaskedImage"_a, "scienceMaskedImage"_a);
        cls.def("getFootprints", &KernelCandidateDetection<PixelT>::getFootprints);
    });
}

}  // namespace lsst::ip::diffim::<anonymous>

void wrapKernelCandidateDetection(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareKernelCandidateDetection<float>(wrappers, "F");
}

}  // diffim
}  // ip
}  // lsst
