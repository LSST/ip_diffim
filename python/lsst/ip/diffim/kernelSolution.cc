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

#include <memory>

#include "Eigen/Core"

#include "lsst/daf/base/PropertySet.h"
#include "lsst/ip/diffim/KernelSolution.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace ip {
namespace diffim {

namespace {

/**
 * Wrap KernelSolution
 */
void declareKernelSolution(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyKernelSolution = py::class_<KernelSolution>;

    wrappers.wrapType(PyKernelSolution(wrappers.module, "KernelSolution"), [](auto &mod, auto &cls) {
        cls.def(py::init<Eigen::MatrixXd, Eigen::VectorXd, bool>(), "mMat"_a, "bVec"_a, "fitForBackground"_a);
        cls.def(py::init<bool>(), "fitForBackground"_a);
        cls.def(py::init<>());

        py::enum_<KernelSolution::KernelSolvedBy>(cls, "KernelSolvedBy")
                .value("NONE", KernelSolution::KernelSolvedBy::NONE)
                .value("CHOLESKY_LDLT", KernelSolution::KernelSolvedBy::CHOLESKY_LDLT)
                .value("CHOLESKY_LLT", KernelSolution::KernelSolvedBy::CHOLESKY_LLT)
                .value("LU", KernelSolution::KernelSolvedBy::LU)
                .value("EIGENVECTOR", KernelSolution::KernelSolvedBy::EIGENVECTOR)
                .export_values();

        py::enum_<KernelSolution::ConditionNumberType>(cls, "ConditionNumberType")
                .value("EIGENVALUE", KernelSolution::ConditionNumberType::EIGENVALUE)
                .value("SVD", KernelSolution::ConditionNumberType::SVD)
                .export_values();

        cls.def("solve", (void (KernelSolution::*)()) &KernelSolution::solve);
        cls.def("solve", (void (KernelSolution::*)(Eigen::MatrixXd const &, Eigen::VectorXd const &)) &
                        KernelSolution::solve,
                "mMat"_a, "bVec"_a);
        cls.def("getSolvedBy", &KernelSolution::getSolvedBy);
        cls.def("getConditionNumber", (double (KernelSolution::*)(KernelSolution::ConditionNumberType)) &
                        KernelSolution::getConditionNumber,
                "conditionType"_a);
        cls.def("getConditionNumber",
                (double (KernelSolution::*)(Eigen::MatrixXd const &, KernelSolution::ConditionNumberType)) &
                        KernelSolution::getConditionNumber,
                "mMat"_a, "conditionType"_a);
        cls.def("getM", &KernelSolution::getM, py::return_value_policy::copy);
        cls.def("getB", &KernelSolution::getB, py::return_value_policy::copy);
        cls.def("printM", &KernelSolution::printM);
        cls.def("printB", &KernelSolution::printB);
        cls.def("printA", &KernelSolution::printA);
        cls.def("getId", &KernelSolution::getId);
    });
}

/**
 * Wrap StaticKernelSolution
 *
 * @tparam InputT  Pixel type of science and template images, e,g, `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Suffix associated with `InputT`, e.g. "F" for `float`
 */
template <typename InputT>
void declareStaticKernelSolution(lsst::cpputils::python::WrapperCollection &wrappers, std::string const &suffix) {
    using PyStaticKernelSolution = py::class_<StaticKernelSolution<InputT>, KernelSolution>;
    std::string name = ("StaticKernelSolution" + suffix);
    wrappers.wrapType(PyStaticKernelSolution(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<lsst::afw::math::KernelList const &, bool>(), "basisList"_a, "fitForBackground"_a);

        cls.def("solve", (void (StaticKernelSolution<InputT>::*)()) &StaticKernelSolution<InputT>::solve);
        cls.def("build", &StaticKernelSolution<InputT>::build, "templateImage"_a, "scienceImage"_a,
                "varianceEstimate"_a);
        cls.def("getKernel", &StaticKernelSolution<InputT>::getKernel);
        cls.def("makeKernelImage", &StaticKernelSolution<InputT>::makeKernelImage);
        cls.def("getBackground", &StaticKernelSolution<InputT>::getBackground);
        cls.def("getKsum", &StaticKernelSolution<InputT>::getKsum);
        cls.def("getSolutionPair", &StaticKernelSolution<InputT>::getSolutionPair);
    });
}

/**
 * Wrap MaskedKernelSolution
 *
 * @tparam InputT  Pixel type of science and template images, e,g, `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Suffix associated with `InputT`, e.g. "F" for `float`
 */
template <typename InputT>
void declareMaskedKernelSolution(lsst::cpputils::python::WrapperCollection &wrappers, std::string const &suffix) {
    using PyMaskedKernelSolution = py::class_<MaskedKernelSolution<InputT>, StaticKernelSolution<InputT>>;
    std::string name = "MaskedKernelSolution" + suffix;
    wrappers.wrapType(PyMaskedKernelSolution(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<lsst::afw::math::KernelList const &, bool>(), "basisList"_a, "fitForBackground"_a);

        cls.def("buildOrig", &MaskedKernelSolution<InputT>::buildOrig, "templateImage"_a, "scienceImage"_a,
                "varianceEstimate"_a, "pixelMask"_a);
        cls.def("buildWithMask", &MaskedKernelSolution<InputT>::buildWithMask, "templateImage"_a,
                "scienceImage"_a, "varianceEstimate"_a, "pixelMask"_a);
        cls.def("buildSingleMaskOrig", &MaskedKernelSolution<InputT>::buildSingleMaskOrig, "templateImage"_a,
                "scienceImage"_a, "varianceEstimate"_a, "maskBox"_a);
    });
}

/**
 * Wrap RegularizedKernelSolution
 *
 * @tparam InputT  Pixel type of science and template images, e,g, `float`
 * @param mod  pybind11 module
 * @param[in] suffix  Suffix associated with `InputT`, e.g. "F" for `float`
 */
template <typename InputT>
void declareRegularizedKernelSolution(lsst::cpputils::python::WrapperCollection &wrappers, std::string const &suffix) {
    using PyRegularizedKernelSolution =
            py::class_<RegularizedKernelSolution<InputT>, StaticKernelSolution<InputT>>;

    std::string name = "RegularizedKernelSolution" + suffix;
    wrappers.wrapType(PyRegularizedKernelSolution(wrappers.module, "RegularizedKernelSolution"), [](auto &mod, auto &cls) {
        cls.def(py::init<lsst::afw::math::KernelList const &, bool, Eigen::MatrixXd const &,
                        daf::base::PropertySet const &>(),
                "basisList"_a, "fitForBackground"_a, "hMat"_a, "ps"_a);

        cls.def("solve",
                (void (RegularizedKernelSolution<InputT>::*)()) &RegularizedKernelSolution<InputT>::solve);
        cls.def("getLambda", &RegularizedKernelSolution<InputT>::getLambda);
        cls.def("estimateRisk", &RegularizedKernelSolution<InputT>::estimateRisk, "maxCond"_a);
        cls.def("getM", &RegularizedKernelSolution<InputT>::getM);
    });
}

/**
 * Wrap SpatialKernelSolution
 */
void declareSpatialKernelSolution(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PySpatialKernelSolution = py::class_<SpatialKernelSolution, KernelSolution> ;

    wrappers.wrapType(PySpatialKernelSolution(wrappers.module, "SpatialKernelSolution"), [](auto &mod, auto &cls) {
        cls.def(py::init<lsst::afw::math::KernelList const &, lsst::afw::math::Kernel::SpatialFunctionPtr,
                        lsst::afw::math::Kernel::SpatialFunctionPtr, daf::base::PropertySet const &>(),
                "basisList"_a, "spatialKernelFunction"_a, "background"_a, "ps"_a);

        cls.def("solve", (void (SpatialKernelSolution::*)()) &SpatialKernelSolution::solve);
        cls.def("addConstraint", &SpatialKernelSolution::addConstraint, "xCenter"_a, "yCenter"_a, "qMat"_a,
                "wVec"_a);
        cls.def("makeKernelImage", &SpatialKernelSolution::makeKernelImage, "pos"_a);
        cls.def("getSolutionPair", &SpatialKernelSolution::getSolutionPair);
    });
}

}  // namespace lsst::ip::diffim::<anonymous>

void wrapKernelSolution(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareKernelSolution(wrappers);
    declareStaticKernelSolution<float>(wrappers, "F");
    declareMaskedKernelSolution<float>(wrappers, "F");
    declareRegularizedKernelSolution<float>(wrappers, "F");
    declareSpatialKernelSolution(wrappers);
}

}  // diffim
}  // ip
}  // lsst
