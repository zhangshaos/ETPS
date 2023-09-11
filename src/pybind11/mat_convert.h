#ifndef CCS_SRC_PYBIND11_MAT_CONVERT_H_
#define CCS_SRC_PYBIND11_MAT_CONVERT_H_

#include <opencv2/core.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


cv::Mat_<cv::Vec3b>
npy_u8c3_to_mat(pybind11::array_t<uint8_t> &npy_mat);

pybind11::array_t<int32_t>
mat_i32_to_npy(cv::Mat_<int> &mat);


#endif //CCS_SRC_PYBIND11_MAT_CONVERT_H_
