#ifndef ETPS_SRC_DRAW_LINE_H_
#define ETPS_SRC_DRAW_LINE_H_

#include <opencv2/core.hpp>
#undef NDEBUG
#include <cassert>


inline
int
float_round(float a) {
  return (int) (a + 0.5f);
}


//https://sanctorum003.github.io/2021/06/05/CG/CG/%E7%94%BB%E7%BA%BF%E7%AE%97%E6%B3%95/
inline
std::vector<cv::Point2i>
line_path(cv::Point2i p0, cv::Point2i p1){
  const int dx = p1.x - p0.x, dy = p1.y - p0.y;
  const int n_step = std::max(std::abs(dx), std::abs(dy));
  if (n_step <= 0)
    return { p0 };
  const float x_inc = (float) dx / (float) n_step,
              y_inc = (float) dy / (float) n_step;
  assert(std::abs(x_inc) >= 1.f || std::abs(y_inc) >= 1.f);

  std::vector<cv::Point2i> result;
  result.reserve(n_step + 1);
  float x = p0.x, y = p0.y;
  result.emplace_back(float_round(x), float_round(y));
  for (int k=n_step; k>0; --k){
    x += x_inc;
    y += y_inc;
    result.emplace_back(float_round(x), float_round(y));
  }
  return result;
}

#endif //ETPS_SRC_DRAW_LINE_H_
