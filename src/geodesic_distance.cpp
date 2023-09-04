#include "geodesic_distance.h"
#include "draw_line.h"


F32
dist_spatial_geodesic(const cv::Point2i &l, const cv::Point2i &r,
                      const cv::Mat_<uchar> &grad){
  constexpr float s = 0.01;
  auto pixels = line_path(l, r);
  assert(l == pixels[0]);
  F32 dist = 0;
  for (int i=1,i_end=pixels.size(); i<i_end; ++i){
    auto p0 = pixels[i-1], p1 = pixels[i];
    int g0 = grad.at<uchar>(p0.y, p0.x),
        g1 = grad.at<uchar>(p1.y, p1.x);
    F32 d = std::abs(g0 - g1) * s + 1.f;
    dist += d;
  }
  return dist*dist;
}