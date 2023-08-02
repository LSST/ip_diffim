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

#include "Eigen/Core"
#include "ndarray/pybind11.h"

#include <memory>
#include <string>

#include "lsst/afw/math/SpatialCell.h"
#include "lsst/ip/diffim/KernelCandidate.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace ip {
namespace diffim {

namespace {

/**
 * Wrap `KernelCandidate` and factory function `makeKernelCandidate` for one pixel type
 *
 * @tparam PixelT  Pixel type of image plane of MaskedImage, e.g. `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Class name suffix associated with PixeT, e.g. "F" for `float`
 */
template <typename PixelT>
void declareKernelCandidate(lsst::cpputils::python::WrapperCollection &wrappers, std::string const &suffix) {
    using CandidateSwitch = typename KernelCandidate<PixelT>::CandidateSwitch;
    using PyCandidateSwitch = py::enum_<CandidateSwitch>;

    std::string name = "KernelCandidate" + suffix;
    using PyKernelCandidate = py::class_<KernelCandidate<PixelT>, afw::math::SpatialCellImageCandidate>;

    auto clsDef = wrappers.wrapType(PyKernelCandidate(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<float const, float const, std::shared_ptr<afw::image::MaskedImage<PixelT>> const &,
                        std::shared_ptr<afw::image::MaskedImage<PixelT>> const &, daf::base::PropertySet const &>(),
                "xCenter"_a, "yCenter"_a, "templateMaskedImage"_a, "scienceMaskedImage"_a, "ps"_a);
        cls.def(py::init<std::shared_ptr<afw::table::SourceRecord> const &,
                        std::shared_ptr<afw::image::MaskedImage<PixelT>> const &,
                        std::shared_ptr<afw::image::MaskedImage<PixelT>> const &, daf::base::PropertySet const &>(),
                "source"_a, "templateMaskedImage"_a, "scienceMaskedImage"_a, "ps"_a);

        cls.def("getCandidateRating", &KernelCandidate<PixelT>::getCandidateRating);
        cls.def("getSource", &KernelCandidate<PixelT>::getSource);
        cls.def("getTemplateMaskedImage", &KernelCandidate<PixelT>::getTemplateMaskedImage);
        cls.def("getScienceMaskedImage", &KernelCandidate<PixelT>::getScienceMaskedImage);
        cls.def("getKernel", &KernelCandidate<PixelT>::getKernel, "cand"_a);
        cls.def("getBackground", &KernelCandidate<PixelT>::getBackground, "cand"_a);
        cls.def("getKsum", &KernelCandidate<PixelT>::getKsum, "cand"_a);
        cls.def("getKernelImage", &KernelCandidate<PixelT>::getKernelImage, "cand"_a);
        cls.def("getImage", &KernelCandidate<PixelT>::getImage);
        cls.def("getKernelSolution", &KernelCandidate<PixelT>::getKernelSolution, "cand"_a);
        cls.def("getDifferenceImage",
                (afw::image::MaskedImage<PixelT> (KernelCandidate<PixelT>::*)(CandidateSwitch)) &
                        KernelCandidate<PixelT>::getDifferenceImage,
                "cand"_a);
        cls.def("getDifferenceImage", (afw::image::MaskedImage<PixelT> (KernelCandidate<PixelT>::*)(
                        std::shared_ptr<afw::math::Kernel>, double)) &
                        KernelCandidate<PixelT>::getDifferenceImage,
                "kernel"_a, "background"_a);
        cls.def("isInitialized", &KernelCandidate<PixelT>::isInitialized);
        cls.def("build", (void (KernelCandidate<PixelT>::*)(afw::math::KernelList const &)) &
                        KernelCandidate<PixelT>::build,
                "basisList"_a);
        cls.def("build",
                (void (KernelCandidate<PixelT>::*)(afw::math::KernelList const &, Eigen::MatrixXd const &)) &
                        KernelCandidate<PixelT>::build,
                "basisList"_a, "hMat"_a);
        mod.def("makeKernelCandidate",
                (std::shared_ptr<KernelCandidate<PixelT>>(*)(
                        float const, float const, std::shared_ptr<afw::image::MaskedImage<PixelT>> const &,
                        std::shared_ptr<afw::image::MaskedImage<PixelT>> const &, daf::base::PropertySet const &)) &
                        makeKernelCandidate,
                "xCenter"_a, "yCenter"_a, "templateMaskedImage"_a, "scienceMaskedImage"_a, "ps"_a);
        mod.def("makeKernelCandidate",
                (std::shared_ptr<KernelCandidate<PixelT>>(*)(
                        std::shared_ptr<afw::table::SourceRecord> const &,
                        std::shared_ptr<afw::image::MaskedImage<PixelT>> const &,
                        std::shared_ptr<afw::image::MaskedImage<PixelT>> const &, daf::base::PropertySet const &)) &
                        makeKernelCandidate,
                "source"_a, "templateMaskedImage"_a, "scienceMaskedImage"_a, "ps"_a);
    });
    wrappers.wrapType(PyCandidateSwitch(clsDef, "CandidateSwitch"), [](auto &mod, auto &enm) {
        enm.value("ORIG", CandidateSwitch::ORIG);
        enm.value("PCA", CandidateSwitch::PCA);
        enm.value("RECENT", CandidateSwitch::RECENT);
        enm.export_values();
    });
}

}  // namespace lsst::ip::diffim::<anonymous>

void wrapKernelCandidate(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareKernelCandidate<float>(wrappers, "F");
}

}  // diffim
}  // ip
}  // lsst
