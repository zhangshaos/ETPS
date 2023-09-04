#ifndef CCS_SRC_CCS_H_
#define CCS_SRC_CCS_H_

#include <opencv2/core.hpp>

struct HyperParams{
  int           expect_spx_num        = 1000;
  mutable float spatial_scale         = 1;//设置初值，之后算法根据expect_spx_num和图片尺寸调整该项
  mutable int   max_iter_num          = 0;//会根据expect_spx_num和图片尺寸而自动选择
  mutable int   min_edge_threshold    = 0;//会根据expect_spx_num和图片尺寸而自动选择
  bool          rgb2lab               = true;
  bool          verbose               = true;
};

cv::Mat_<int>
ccs(const cv::Mat_<cv::Vec3b> &rgb_img, const HyperParams &params);

#endif //CCS_SRC_CCS_H_
