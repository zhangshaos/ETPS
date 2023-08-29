#ifndef FH_SRC_FH_H_
#define FH_SRC_FH_H_

#include <opencv2/core.hpp>

struct HyperParams{
  float k = 800;
  float visualize_step_ratio = 0.1f;
  int radius   = 1;
  float k_sigma_ratio = 2.f;
  bool rgb2lab = true;
  bool verbose = true;

  mutable float _max_edge_weight = 0;
  mutable float _max_num_vertex = 0;
};

cv::Mat_<int>
fh(const cv::Mat_<cv::Vec3b> &rgb_img, const HyperParams &params);

#endif //FH_SRC_FH_H_
