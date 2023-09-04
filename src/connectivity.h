#ifndef CCS_SRC_CONNECTIVITY_H_
#define CCS_SRC_CONNECTIVITY_H_

#include "ccs.h"
#include <opencv2/core.hpp>


cv::Mat_<int>
check_connectivity(const cv::Mat_<int> &label,
                   const cv::Mat_<uchar> &edge,
                   const HyperParams &params);


cv::Mat_<uchar>
edge_mat(const cv::Mat_<uchar> &grad,
         const HyperParams &params);


#endif //CCS_SRC_CONNECTIVITY_H_
