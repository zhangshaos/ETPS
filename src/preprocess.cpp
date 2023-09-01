#include "preprocess.h"
#include <opencv2/imgproc.hpp>
#define D_RGB2GRAD 0
#if D_RGB2GRAD
#include "print_cv_mat.h"
#include <opencv2/imgcodecs.hpp>
#endif


cv::Mat_<cv::Vec3b>
rgb2lab(const cv::Mat_<cv::Vec3b> &rgb_img) {
  cv::Mat_<cv::Vec3b> lab_img;
  cv::cvtColor(rgb_img, lab_img, cv::COLOR_RGB2Lab);
  for (int y = rgb_img.rows - 1; y >= 0; --y)
    for (int x = rgb_img.cols - 1; x >= 0; --x) {
      float l = lab_img.at<cv::Vec3b>(y, x)[0];
      l = std::clamp((l / 255.f * 100.f), 0.f, 100.f);
      lab_img.at<cv::Vec3b>(y, x)[0] = (uchar) l;
    }
  return lab_img;
}

cv::Mat_<cv::Vec3b>
bgr2lab(const cv::Mat_<cv::Vec3b> &bgr_img) {
  cv::Mat_<cv::Vec3b> lab_img;
  cv::cvtColor(bgr_img, lab_img, cv::COLOR_BGR2Lab);
  for (int y = bgr_img.rows - 1; y >= 0; --y)
    for (int x = bgr_img.cols - 1; x >= 0; --x) {
      float l = lab_img.at<cv::Vec3b>(y, x)[0];
      l = std::clamp((l / 255.f * 100.f), 0.f, 100.f);
      lab_img.at<cv::Vec3b>(y, x)[0] = (uchar) l;
    }
  return lab_img;
}


//blur_sigma=0 means automatic selection by cv::GaussianBlur
static
cv::Mat_<uchar>
canny(const cv::Mat_<uchar> &src, bool non_max_suppress=true, double blur_sigma=0) {
  cv::Mat gauss_src;
  cv::GaussianBlur(src, gauss_src, cv::Size(5, 5), blur_sigma);
  CV_Assert(gauss_src.type() == CV_8U);

  cv::Mat dx_mat, dy_mat;
  cv::Scharr(gauss_src, dx_mat, CV_16S, 1, 0, 1, 0, cv::BORDER_REPLICATE);
  cv::Scharr(gauss_src, dy_mat, CV_16S, 0, 1, 1, 0, cv::BORDER_REPLICATE);
  CV_Assert(dx_mat.type() == CV_16S);
  CV_Assert(dy_mat.type() == CV_16S);

  cv::Mat_<float> angle(dx_mat.rows, dx_mat.cols, 0.f);
  cv::Mat_<uchar> result(dx_mat.rows, dx_mat.cols, (uchar) 0);
  for (int y = 0; y < angle.rows; ++y)
    for (int x = 0; x < angle.cols; ++x) {
      float dx = dx_mat.at<short>(y, x),
          dy = dy_mat.at<short>(y, x);
      angle.at<float>(y, x) = std::atan2(dy, dx);
      float r = std::sqrt(dx * dx + dy * dy);
      result.at<uchar>(y, x) = (uchar) std::clamp(r, 0.f, 255.f);
    }

  if (!non_max_suppress)
    return result;

  constexpr float PI_8 = M_PI_4 * 0.5f, //22.5
  PI_3_8 = PI_8 * 3,      //67.5
  PI_5_8 = PI_8 * 5,      //112.5
  PI_7_8 = PI_8 * 7,      //157.5
  PI     = PI_8 * 8;

  //非极大值抑制
  const cv::Mat_<uchar> nms_src_mat = result.clone();
  for (int y = 0; y < nms_src_mat.rows; ++y)
    for (int x = 0; x < nms_src_mat.cols; ++x) {
      float a = angle.at<float>(y, x);
      CV_Assert(-PI <= a && a <= PI);
      uchar v = nms_src_mat.at<uchar>(y, x);
      if ((-PI_8 < a) && (a <= PI_8) ||
          (PI_7_8 < a) || (a <= -PI_7_8)) {
        //Horizontal Edge
        if ((x < nms_src_mat.cols - 1) &&
            (v < nms_src_mat.at<uchar>(y, x+1)) ||
            (x >= 1) &&
            (v < nms_src_mat.at<uchar>(y, x-1)))
          result.at<uchar>(y, x) = 0;
      } else if ((-PI_5_8 < a) && (a <= -PI_3_8) ||
          (PI_3_8 < a) && (a <= PI_5_8)) {
        //Vertical Edge
        if ((y < nms_src_mat.rows - 1) &&
            (v < nms_src_mat.at<uchar>(y+1, x)) ||
            (y >= 1) &&
            (v < nms_src_mat.at<uchar>(y-1, x)))
          result.at<uchar>(y, x) = 0;
      } else if ((-PI_3_8 < a) && (a <= -PI_8) ||
          (PI_5_8 < a) && (a <= PI_7_8)) {
        //-45 Degree Edge
        if ((y >= 1 && x < nms_src_mat.cols - 1) &&
            (v < nms_src_mat.at<uchar>(y-1, x+1)) ||
            (y < nms_src_mat.rows - 1 && x >= 1) &&
            (v < nms_src_mat.at<uchar>(y+1, x-1)))
          result.at<uchar>(y, x) = 0;
      } else if ((-PI_7_8 < a) && (a <= -PI_5_8) ||
          (PI_8 < a) && (a <= PI_3_8)) {
        //45 Degree Edge
        if ((y < nms_src_mat.rows - 1 && x < nms_src_mat.cols - 1) &&
            (v < nms_src_mat.at<uchar>(y+1, x+1)) ||
            (y >= 1 && x >= 1) &&
            (v < nms_src_mat.at<uchar>(y-1, x-1)))
          result.at<uchar>(y, x) = 0;
      } else {
        CV_Assert(0 && "impossible!");
      }
    }
  return result;
}


cv::Mat_<uchar>
rgb2grad(const cv::Mat_<cv::Vec3b> &rgb_img) {
#if D_RGB2GRAD
  {
  {
    cv::Mat t0(10, 10, CV_16S);
    cv::Mat_<short> t1(10, 10);
    printf("%s == %s\n",
           cv::typeToString(t0.type()).c_str(),
           cv::typeToString(t1.type()).c_str());
    fflush(stdout);
  }
  cv::Mat_<uchar> gray_img;
  cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);

  cv::Mat lap_grad, sch_grad_x, sch_grad_y;
  cv::Laplacian(gray_img, lap_grad, CV_16S, 3);
  cv::Scharr(gray_img, sch_grad_x, CV_16S, 1, 0);
  cv::Scharr(gray_img, sch_grad_y, CV_16S, 0, 1);
  printf("%s, %s, %s\n",
         cv::typeToString(lap_grad.type()).c_str(),
         cv::typeToString(sch_grad_x.type()).c_str(),
         cv::typeToString(sch_grad_y.type()).c_str());
  fflush(stdout);

  cv::Mat lap_img, sch_img;
  cv::convertScaleAbs(lap_grad, lap_img);
  cv::Mat sch_x, sch_y;
  cv::convertScaleAbs(sch_grad_x, sch_x);
  cv::convertScaleAbs(sch_grad_y, sch_y);
  cv::addWeighted(sch_x, 0.5, sch_y, 0.5, 0, sch_img);
  printf("%s, %s\n",
         cv::typeToString(lap_img.type()).c_str(),
         cv::typeToString(sch_img.type()).c_str());
  fflush(stdout);

  cv::Mat lab_img = rgb2lab(rgb_img);
  cv::Mat lab[3];
  cv::split(lab_img, lab);
  auto canny_img_l = canny(lab[0]);
  auto canny_img_a = canny(lab[1]);
  auto canny_img_b = canny(lab[2]);
  cv::Mat canny_img_ab;
  cv::addWeighted(canny_img_a, 1, canny_img_b, 1, 0, canny_img_ab, CV_8U);

  cv::Mat edge_map;
  cv::Canny(gray_img, edge_map, 25, 100, 3, true);
  auto canny_img = canny(gray_img);

  cv::imwrite("lap.png", lap_img);
  cv::imwrite("scharr.png", sch_img);
  cv::imwrite("canny.png", canny_img);
  cv::imwrite("canny_l.png", canny_img_l);
  cv::imwrite("canny_a.png", canny_img_a);
  cv::imwrite("canny_b.png", canny_img_b);
  cv::imwrite("canny_ab.png", canny_img_ab);
  cv::imwrite("edge.png", edge_map);
  }
#endif
  auto lab_img = rgb2lab(rgb_img);
  cv::Mat_<uchar> lab[3];
  cv::split(lab_img, lab);
  //Future: merge l* a* b* result will be better?
  auto result = canny(lab[0]);
  return result;
}
