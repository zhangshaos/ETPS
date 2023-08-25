#ifndef TASP_PRINT_CV_MAT_H
#define TASP_PRINT_CV_MAT_H

#include <string>
#include <opencv2/core.hpp>


std::string
str_printf(size_t buf_size, const char *const fmt, ...);

std::string
str_cv_mat(const cv::Mat_<int> &mat);


bool
save_mat(const std::string &file_path,
         const std::string &content);

bool
save_image(const std::string &file_path,
           const cv::Mat_<cv::Vec3b> &img,
           bool bgr2rgb=true);

bool
save_image(const std::string &file_path,
           const cv::Mat_<uchar> &img);

bool
save_edge_map(const std::string &file_path,
              const cv::Mat_<int> &mat);


void
sample_color(double *color, double x, double min, double max);


bool
save_segmentation_map(const std::string &file_path,
                      const cv::Mat_<int> &mat);


cv::Mat_<cv::Vec3b>
mark_boundaries(const cv::Mat_<cv::Vec3b> &rgb_img,
                const cv::Mat_<int> &labels,
                const cv::Vec3b &border_rgb);


bool
save_image_with_segmentation(const std::string &file_path,
                             const cv::Mat_<cv::Vec3b> &img,
                             const cv::Mat_<int> &mat);


bool
save_super_pixel(const std::string &file_path,
                 const std::vector<cv::Point2i> &spx_s,
                 const cv::Mat_<int> &labels,
                 const cv::Mat_<cv::Vec3b> &rgb_img);


#endif //TASP_PRINT_CV_MAT_H
