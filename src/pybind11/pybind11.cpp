#include <pybind11/pybind11.h>
#include "ccs.h"
#include "mat_convert.h"


pybind11::array_t<int32_t>
ccs_segment(pybind11::array_t<uint8_t> &rgb_img,
            int expect_spx_num=1000,
            float spatial_scale=0.3f,
            bool rgb2lab=true,
            bool verbose=true){
  auto _rgb_img = npy_u8c3_to_mat(rgb_img);
  HyperParams params;
  params.expect_spx_num = expect_spx_num;
  params.spatial_scale  = spatial_scale;
  params.rgb2lab        = rgb2lab;
  params.verbose        = verbose;
  auto mat = ccs(_rgb_img, params);
  auto res = mat_i32_to_npy(mat);
  return res;
}


PYBIND11_MODULE(pyCCS, m){
  m.doc() = "CCS super-pixel segmentation algorithm.";
  m.def("ccs", &ccs_segment, "CCS super-pixel segmentation algorithm.",
        pybind11::arg("rgb_img"),
        pybind11::arg("expect_spx_num")=1000,
        pybind11::arg("spatial_scale")=0.3f,
        pybind11::arg("rgb2lab")=true,
        pybind11::arg("verbose")=true);
}