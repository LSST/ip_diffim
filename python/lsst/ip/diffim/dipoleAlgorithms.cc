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

#include "lsst/ip/diffim/DipoleAlgorithms.h"
#include "lsst/meas/base/Algorithm.h"
#include "lsst/pex/config/python.h"  // for LSST_DECLARE_CONTROL_FIELD

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace ip {
namespace diffim {

namespace {

void declareDipoleCentroidControl(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyDipoleCentroidControl = py::class_<DipoleCentroidControl>;

    wrappers.wrapType(PyDipoleCentroidControl(wrappers.module, "DipoleCentroidControl"), [](auto &mod, auto &cls) {
        cls.def(py::init<>());
    });
}

void declareDipoleFluxControl(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyDipoleFluxControl = py::class_<DipoleFluxControl>;

    wrappers.wrapType(PyDipoleFluxControl(wrappers.module, "DipoleFluxControl"), [](auto &mod, auto &cls) {
        cls.def(py::init<>());
    });
}

void declareDipolePsfFluxControl(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyPsfDipoleFluxControl =
            py::class_<PsfDipoleFluxControl, DipoleFluxControl>;
    wrappers.wrapType(PyPsfDipoleFluxControl(wrappers.module, "PsfDipoleFluxControl"), [](auto &mod, auto &cls) {
        cls.def(py::init<>());

        LSST_DECLARE_CONTROL_FIELD(cls, PsfDipoleFluxControl, stepSizeCoord);
        LSST_DECLARE_CONTROL_FIELD(cls, PsfDipoleFluxControl, stepSizeFlux);
        LSST_DECLARE_CONTROL_FIELD(cls, PsfDipoleFluxControl, errorDef);
        LSST_DECLARE_CONTROL_FIELD(cls, PsfDipoleFluxControl, maxFnCalls);
    });
}

void declareDipoleCentroidAlgorithm(lsst::cpputils::python::WrapperCollection &wrappers) {
    // Abstract class, so add a leading underscore to Python name and do not wrap constructor
    using PyDipoleCentroidAlgorithm =
            py::class_<DipoleCentroidAlgorithm, meas::base::SimpleAlgorithm>;

    wrappers.wrapType(PyDipoleCentroidAlgorithm(wrappers.module, "_DipoleCentroidAlgorithm"), [](auto &mod, auto &cls) {
        cls.attr("FAILURE") = py::cast(DipoleCentroidAlgorithm::FAILURE);
        cls.attr("POS_FLAG") = py::cast(DipoleCentroidAlgorithm::POS_FLAG);
        cls.attr("NEG_FLAG") = py::cast(DipoleCentroidAlgorithm::NEG_FLAG);
        cls.def_static("getFlagDefinitions", &DipoleCentroidAlgorithm::getFlagDefinitions,
                       py::return_value_policy::copy);

        cls.def("getPositiveKeys", &DipoleCentroidAlgorithm::getPositiveKeys);
        cls.def("getNegativeKeys", &DipoleCentroidAlgorithm::getNegativeKeys);
    });
}

void declareDipoleFluxAlgorithm(lsst::cpputils::python::WrapperCollection &wrappers) {
    // Abstract class, so add a leading underscore to Python name and do not wrap constructor
    using PyDipoleFluxAlgorithm =
            py::class_<DipoleFluxAlgorithm, meas::base::SimpleAlgorithm>;
    wrappers.wrapType(PyDipoleFluxAlgorithm(wrappers.module, "_DipoleFluxAlgorithm"), [](auto &mod, auto &cls) {
        cls.attr("FAILURE") = py::cast(DipoleFluxAlgorithm::FAILURE);
        cls.attr("POS_FLAG") = py::cast(DipoleFluxAlgorithm::POS_FLAG);
        cls.attr("NEG_FLAG") = py::cast(DipoleFluxAlgorithm::NEG_FLAG);
        cls.def_static("getFlagDefinitions", &DipoleFluxAlgorithm::getFlagDefinitions,
                       py::return_value_policy::copy);

        cls.def("getPositiveKeys", &DipoleFluxAlgorithm::getPositiveKeys);
        cls.def("getNegativeKeys", &DipoleFluxAlgorithm::getNegativeKeys);
    });
}

void declareNaiveDipoleFlux(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyNaiveDipoleFlux = py::class_<NaiveDipoleFlux, DipoleFluxAlgorithm>;

    wrappers.wrapType(PyNaiveDipoleFlux(wrappers.module, "NaiveDipoleFlux"), [](auto &mod, auto &cls) {
        cls.def(py::init<NaiveDipoleFlux::Control const &, std::string const &, afw::table::Schema &>(), "ctrl"_a,
                "name"_a, "schema"_a);

        cls.def("measure", &NaiveDipoleFlux::measure, "measRecord"_a, "exposure"_a);
        cls.def("fail", &NaiveDipoleFlux::fail, "measRecord"_a, "error"_a = nullptr);
    });
}

void declareNaiveDipoleCentroid(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyNaiveDipoleCentroid =
            py::class_<NaiveDipoleCentroid, DipoleCentroidAlgorithm>;
    wrappers.wrapType(PyNaiveDipoleCentroid(wrappers.module, "NaiveDipoleCentroid"), [](auto &mod, auto &cls) {
        cls.def(py::init<NaiveDipoleCentroid::Control const &, std::string const &, afw::table::Schema &>(),
                "ctrl"_a, "name"_a, "schema"_a);

        cls.def("getCenterKeys", &NaiveDipoleCentroid::getCenterKeys);
        cls.def("getPositiveKeys", &NaiveDipoleCentroid::getPositiveKeys);
        cls.def("getNegativeKeys", &NaiveDipoleCentroid::getNegativeKeys);

        cls.def("measure", &NaiveDipoleCentroid::measure, "measRecord"_a, "exposure"_a);
        cls.def("mergeCentroids", &NaiveDipoleCentroid::mergeCentroids, "source"_a, "posValue"_a, "negValue"_a);
        cls.def("fail", &NaiveDipoleCentroid::fail, "measRecord"_a, "error"_a = nullptr);
    });
}

void declarePsfDipoleFlux(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyPsfDipoleFlux =  py::class_<PsfDipoleFlux, DipoleFluxAlgorithm>;

    wrappers.wrapType(PyPsfDipoleFlux(wrappers.module, "PsfDipoleFlux"), [](auto &mod, auto &cls) {
        cls.def(py::init<PsfDipoleFlux::Control const &, std::string const &, afw::table::Schema &>(), "ctrl"_a,
                "name"_a, "schema"_a);

        cls.def("chi2", &PsfDipoleFlux::chi2, "source"_a, "exposure"_a, "negCenterX"_a, "negCenterY"_a,
                "negFlux"_a, "posCenterX"_a, "posCenterY"_a, "posFlux"_a);
        cls.def("measure", &PsfDipoleFlux::measure, "measRecord"_a, "exposure"_a);
        cls.def("fail", &PsfDipoleFlux::fail, "measRecord"_a, "error"_a = nullptr);
    });
}

}  // namespace lsst::ip::diffim::<anonymous>

void wrapDipoleAlgorithms(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareDipoleCentroidControl(wrappers);
    declareDipoleFluxControl(wrappers);
    declareDipolePsfFluxControl(wrappers);
    declareDipoleCentroidAlgorithm(wrappers);
    declareDipoleFluxAlgorithm(wrappers);
    declareNaiveDipoleFlux(wrappers);
    declareNaiveDipoleCentroid(wrappers);
    declarePsfDipoleFlux(wrappers);
}

}  // diffim
}  // ip
}  // lsst
