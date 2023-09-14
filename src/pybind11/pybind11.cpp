#include <pybind11/pybind11.h>
#include "ccs.h"
#include "ccs_semantic_seg.h"
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


pybind11::array_t<int32_t>
naive_semantic_segment(pybind11::array_t<int32_t> &spx_label,
                       pybind11::array_t<uint8_t> &rgb_img,
                       pybind11::array_t<float> &soft_semantic_score){
  auto spx_label_   = npy_i32_to_mat(spx_label);
  auto rgb_img_     = npy_u8c3_to_mat(rgb_img);
  auto logits       = npy_f32cn_to_mat(soft_semantic_score);
  auto mat = naive_semantic_seg(spx_label_, rgb_img_, logits);
  auto res = mat_i32_to_npy(mat);
  return res;
}


pybind11::array_t<int32_t>
crf_semantic_segment(pybind11::array_t<int32_t> &spx_label,
                     pybind11::array_t<uint8_t> &rgb_img,
                     pybind11::array_t<float> &soft_semantic_score,
                     float wi=10,
                     float wj=1,
                     bool verbose=true){
  auto spx_label_   = npy_i32_to_mat(spx_label);
  auto rgb_img_     = npy_u8c3_to_mat(rgb_img);
  auto logits       = npy_f32cn_to_mat(soft_semantic_score);
  CRF_Params params;
  params.wi         = wi;
  params.wj         = wj;
  params.verbose    = verbose;
  auto mat = crf_semantic_seg(spx_label_, rgb_img_, logits, params);
  auto res = mat_i32_to_npy(mat);
  return res;
}


pybind11::array_t<int32_t>
mrf_semantic_segment(pybind11::array_t<int32_t> &spx_label,
                     pybind11::array_t<uint8_t> &rgb_img,
                     pybind11::array_t<float> &soft_semantic_score,
                     float wi=10,
                     float wj=1,
                     bool verbose=true){
  auto spx_label_   = npy_i32_to_mat(spx_label);
  auto rgb_img_     = npy_u8c3_to_mat(rgb_img);
  auto logits       = npy_f32cn_to_mat(soft_semantic_score);
  MRF_Params params;
  params.wi         = wi;
  params.wj         = wj;
  params.verbose    = verbose;
  auto mat = mrf_semantic_seg(spx_label_, rgb_img_, logits, params);
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
  m.def("naive_segment", &naive_semantic_segment,
        "NAIVE semantic segment algorithm based on super-pixel",
        pybind11::arg("spx_label"),
        pybind11::arg("rgb_image"),
        pybind11::arg("semantic_logits"));
  m.def("crf_segment", &crf_semantic_segment,
        "CRF semantic segment algorithm based on super-pixel",
        pybind11::arg("spx_label"),
        pybind11::arg("rgb_image"),
        pybind11::arg("semantic_logits"),
        pybind11::arg("wi")=10,
        pybind11::arg("wj")=1,
        pybind11::arg("verbose")=true);
  m.def("mrf_segment", &mrf_semantic_segment,
        "MRF semantic segment algorithm based on super-pixel",
        pybind11::arg("spx_label"),
        pybind11::arg("rgb_image"),
        pybind11::arg("semantic_logits"),
        pybind11::arg("wi")=10,
        pybind11::arg("wj")=1,
        pybind11::arg("verbose")=true);
}