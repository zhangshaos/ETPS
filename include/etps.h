#ifndef ETPS_SRC_ETPS_H_
#define ETPS_SRC_ETPS_H_

#include <opencv2/core.hpp>

struct HyperParams{
  int expect_spx_num = 1000;
  int max_iter_num_in_each_level = 2;
  float spatial_scale = 1;
  bool rgb2lab = true;
  bool verbose = true;

  mutable float _max_rgb_dist = 0;
  mutable float _max_spatial_dist = 0;
  mutable float _record_max_rgb_dist = 0;
  mutable float _record_max_spatial_dist = 0;
  void
  update_record()const{
    _max_rgb_dist = _record_max_rgb_dist;
    _max_spatial_dist = _record_max_spatial_dist;
    _record_max_rgb_dist = 0;
    _record_max_spatial_dist = 0;
  }
};

cv::Mat_<int>
etps(const cv::Mat_<cv::Vec3b> &rgb_img, const HyperParams &params);

#endif //ETPS_SRC_ETPS_H_
