#include "mat_convert.h"


cv::Mat_<cv::Vec3b>
npy_u8c3_to_mat(pybind11::array_t<uint8_t> &npy_mat){
  if (npy_mat.ndim() != 3)
    throw std::runtime_error("3-channel image must be 3 dims!");
  pybind11::buffer_info buf = npy_mat.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
  return mat;
}


pybind11::array_t<int32_t>
mat_i32_to_npy(cv::Mat_<int> &mat){
  pybind11::array_t<int32_t> res({ mat.rows, mat.cols }, (int32_t*)mat.data);
  return res;
}
