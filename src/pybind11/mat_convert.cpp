#include "mat_convert.h"


cv::Mat_<cv::Vec3b>
npy_u8c3_to_mat(pybind11::array_t<uint8_t> &npy_mat){
  if (npy_mat.ndim() != 3)
    throw std::runtime_error(".npy must be (H,W,3)!");
  pybind11::buffer_info buf = npy_mat.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
  return mat;
}


pybind11::array_t<int32_t>
mat_i32_to_npy(cv::Mat_<int> &mat){
  pybind11::array_t<int32_t> res({ mat.rows, mat.cols }, (int32_t*)mat.data);
  return res;
}


cv::Mat_<float>
npy_f32cn_to_mat(pybind11::array_t<float> &npy_mat){
  if (npy_mat.ndim() != 3)
    throw std::runtime_error(".npy must be (H,W,C)!");
  pybind11::buffer_info buf = npy_mat.request();
  int shape[3] = { (int)npy_mat.shape(0),
                   (int)npy_mat.shape(1),
                   (int)npy_mat.shape(2) };
  cv::Mat mat(3, shape, CV_32F, buf.ptr);
  return mat;
}


cv::Mat_<int>
npy_i32_to_mat(pybind11::array_t<int32_t> &npy_mat){
  pybind11::buffer_info buf = npy_mat.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], CV_32S, buf.ptr);
  return mat;
}
