#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "headers/murty.hpp"
#include "headers/MurtyMiller.hpp"
#include "headers/pd.hpp"
#include "headers/gibbs.hpp"
#include "headers/bbox.hpp"
#include "headers/esf.hpp"
#include "headers/utils.hpp"
#include "joint_glmb/joint_glmb.hpp"
#include "joint_lmb/joint_lmb.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpputils, m) {
    // Murty
    py::class_<lap::Murty>(m, "Murty")
        .def(py::init<>())
        .def(py::init<lap::CostMatrix>())
        .def("draw", &lap::Murty::draw_tuple)
        .def("draw_solutions", &lap::Murty::draw_solutions);
    m.def("MurtyMiller", &MurtyMiller<double>::getKBestAssignments);

    // RFS Joint GLMB
    py::class_<GLMB>(m, "GLMB")
        .def(py::init<int, int, bool, bool>())
        .def("run_glmb_feat", &GLMB::run_glmb_feat)
        .def("run_glmb", &GLMB::run_glmb);

    // RFS JointGLMB
    py::class_<LMB>(m, "LMB")
        .def(py::init<int, int>())
        .def("run_lmb_feat", &LMB::run_lmb_feat)
        .def("run_lmb", &LMB::run_lmb);

    // Detection Probability
    py::class_<ComputePD>(m, "ComputePD")
        .def(py::init<std::string>())
        .def("set_recompute_cost", &ComputePD::set_recompute_cost)
        .def("recompute_cost", &ComputePD::recompute_cost)
        .def("compute", &ComputePD::compute);

    // Gibbs Sampling
    m.def("gibbs_jointpredupdt", &gibbs_jointpredupdt);

    // Bounding boxes overlap
    m.def("bboxes_ioi_xyah_back2front", &bboxes_ioi_xyah_back2front);
    m.def("bboxes_ioi_xyah_back2front_all", &bboxes_ioi_xyah_back2front_all);
    m.def("bbox_iou_xyah", &bbox_iou_xyah);

    // ESF : Calculate elementary symmetric function
    m.def("esf", &esf);
    m.def("log_esf", &log_esf);
}