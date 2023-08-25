#include "print_cv_mat.h"
#include <vector>
#include <fstream>
#include <cstdarg>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


std::string
str_printf(size_t buf_size, const char *const fmt, ...){
  //Copied from https://blog.csdn.net/xcoderone/article/details/48735425
  char *buf = (char*) alloca(buf_size);
  std::fill(buf, buf+buf_size, '\0');
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, buf_size, fmt, args);
  va_end(args);
  std::string s = buf;
  return s;
}


std::string
str_cv_mat(const cv::Mat_<int> &mat){
  std::vector<std::string> str_mat(mat.rows * mat.cols);
  int max_str_len = 0;
  for (int y=0; y<mat.rows; ++y)
    for (int x = 0; x < mat.cols; ++x) {
      auto s = std::to_string(mat.at<int>(y, x));
      max_str_len = s.size() > max_str_len ? (int) s.size() : max_str_len;
      int i = y * mat.cols + x;
      str_mat[i] = std::move(s);
    }
  std::string s;
  s.reserve(2 * str_mat.size());
  for (int y=0; y<mat.rows; ++y) {
    for (int x = 0; x < mat.cols; ++x) {
      int i = y * mat.cols + x;
      s.append(str_mat[i]);
      int n_space = (max_str_len + 1 - str_mat[i].size());
      assert(n_space > 0);
      std::string t(n_space, ' ');
      t.front() = ',';
      s.append(t);
    }
    s.append("\n");
  }
  return s;
}


bool
save_image(const std::string &file_path,
           const cv::Mat_<cv::Vec3b> &img,
           bool bgr2rgb){
  if (bgr2rgb) {
    cv::Mat_<cv::Vec3b> rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    return cv::imwrite(file_path, rgb_img);
  } else {
    return cv::imwrite(file_path, img);
  }
}


bool
save_image(const std::string &file_path,
           const cv::Mat_<uchar> &img){
  return cv::imwrite(file_path, img);
}


bool
save_mat(const std::string &file_path,
         const std::string &content){
  std::ofstream f;
  f.open(file_path);
  if (f.is_open()) {
    f.write(content.c_str(), content.size());
    f.flush();
    f.close();
    return true;
  }
  return false;
}


inline
cv::Mat_<uchar>
find_boundary(const cv::Mat_<int> &mat){
  cv::Mat_<uchar> border(mat.rows, mat.cols, uchar(0));
  constexpr int offset[][2] = {
      {-1, -1},
      { 0, -1},
      { 1, -1},
      {-1,  0},
      { 1,  0},
      {-1,  1},
      { 0,  1},
      { 1,  1},
  };

  for (int y=0; y<mat.rows; ++y)
    for (int x=0; x<mat.cols; ++x){
      const int v0 = mat.at<int>(y, x);
      bool diff = false;
      for (int j=std::size(offset)-1; j>=0; --j){
        int y1 = y + offset[j][1], x1 = x + offset[j][0];
        if (y1 >= 0 && y1 < mat.rows &&
            x1 >= 0 && x1 < mat.cols){
          int v1 = mat.at<int>(y1, x1);
          if (v1 != v0) {
            diff = true;
            break;
          }
        }
      }
      if (diff)
        border.at<uchar>(y, x) = 0xff;
    }

  //make the border more thinner
  for (int y=0; y<mat.rows; ++y)
    for (int x=0; x<mat.cols; ++x){
      if (!border.at<uchar>(y, x))
        continue;
      bool pass_if_removed = false;
      const int v0 = mat.at<int>(y, x);
      for (int j=std::size(offset)-1; j>=0; --j){
        int y1 = y + offset[j][1], x1 = x + offset[j][0];
        if (y1 >= 0 && y1 < mat.rows &&
            x1 >= 0 && x1 < mat.cols &&
            !border.at<uchar>(y1, x1)){
          int v1 = mat.at<int>(y1, x1);
          if (v1 != v0) {
            pass_if_removed = true;
            break;
          }
        }
      }
      if (!pass_if_removed)
        border.at<uchar>(y, x) = 0;
    }

  //future：让面积大的收缩边界，面积小的边界不变？
  return border;
}

bool
save_edge_map(const std::string &file_path,
              const cv::Mat_<int> &mat){
  auto border = find_boundary(mat);
  return cv::imwrite(file_path, border);
}


void
sample_color(double *color, double x, double min, double max) {
  /*
   * Red = 0
   * Green = 1
   * Blue = 2
   */
  double posSlope = (max - min) / 60;
  double negSlope = (min - max) / 60;

  if (x < 60) {
    color[0] = max;
    color[1] = posSlope * x + min;
    color[2] = min;
    return;
  } else if (x < 120) {
    color[0] = negSlope * x + 2 * max + min;
    color[1] = max;
    color[2] = min;
    return;
  } else if (x < 180) {
    color[0] = min;
    color[1] = max;
    color[2] = posSlope * x - 2 * max + min;
    return;
  } else if (x < 240) {
    color[0] = min;
    color[1] = negSlope * x + 4 * max + min;
    color[2] = max;
    return;
  } else if (x < 300) {
    color[0] = posSlope * x - 4 * max + min;
    color[1] = min;
    color[2] = max;
    return;
  } else {
    color[0] = max;
    color[1] = min;
    color[2] = negSlope * x + 6 * max;
    return;
  }
}


bool
save_segmentation_map(const std::string &file_path,
                      const cv::Mat_<int> &mat){
  cv::Mat_<cv::Vec3b> img(mat.rows, mat.cols);;
  double min_v=0, max_v=0;
  cv::minMaxIdx(mat, &min_v, &max_v);
  for (int y=0; y<mat.rows; ++y)
    for (int x=0; x<mat.cols; ++x){
      double rgb[3] = {0,0,0};
      double v = mat.at<int>(y, x);
      sample_color(rgb, v, min_v, max_v);
      cv::Vec3b c(uchar(255*rgb[0]),
                  uchar(255*rgb[1]),
                  uchar(255*rgb[2]));
      img.at<cv::Vec3b>(y, x) = c;
    }
  return save_image(file_path, img, true);
}


cv::Mat_<cv::Vec3b>
mark_boundaries(const cv::Mat_<cv::Vec3b> &rgb_img,
                const cv::Mat_<int> &labels,
                const cv::Vec3b &border_rgb){
  auto out_img = rgb_img.clone();
  cv::Mat_<uchar> border = find_boundary(labels);
  for (int y=0; y<rgb_img.rows; ++y)
    for (int x=0; x<rgb_img.cols; ++x){
      if (!border.at<uchar>(y, x))
        continue;
      out_img.at<cv::Vec3b>(y, x) = border_rgb;
    }
  return out_img;
}


bool save_image_with_segmentation(const std::string &file_path,
                                  const cv::Mat_<cv::Vec3b> &img,
                                  const cv::Mat_<int> &mat) {
  auto ans = mark_boundaries(img, mat, cv::Vec3b(0xff, 0, 0));
  return save_image(file_path, ans, true);
}


bool
save_super_pixel(const std::string &file_path,
                 const std::vector<cv::Point2i> &spx_s,
                 const cv::Mat_<int> &labels,
                 const cv::Mat_<cv::Vec3b> &rgb_img){
  auto ans = mark_boundaries(rgb_img, labels, cv::Vec3b(0xff, 0, 0));
  for (int j=spx_s.size()-1; j>=0; --j)
    cv::drawMarker(ans, spx_s[j], cv::Scalar(0, 255, 0), cv::MARKER_STAR);
  return save_image(file_path, ans, true);
}