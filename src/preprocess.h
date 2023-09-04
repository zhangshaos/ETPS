#ifndef TASP_SRC_PREPROCESS_H_
#define TASP_SRC_PREPROCESS_H_

#include <opencv2/core.hpp>


cv::Mat_<cv::Vec3b>
rgb2lab(const cv::Mat_<cv::Vec3b> &rgb_img, bool scale);

cv::Mat_<cv::Vec3b>
bgr2lab(const cv::Mat_<cv::Vec3b> &bgr_img);

cv::Mat_<uchar>
rgb2grad(const cv::Mat_<cv::Vec3b> &gray_img);

#endif //TASP_SRC_PREPROCESS_H_
