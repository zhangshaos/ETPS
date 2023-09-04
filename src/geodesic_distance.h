#ifndef CCS_SRC_GEODESIC_DISTANCE_H_
#define CCS_SRC_GEODESIC_DISTANCE_H_

#include "self_check_float.h"
#include <opencv2/core.hpp>

F32
dist_spatial_geodesic(const cv::Point2i &l, const cv::Point2i &r,
                      const cv::Mat_<uchar> &grad);

#endif //CCS_SRC_GEODESIC_DISTANCE_H_
