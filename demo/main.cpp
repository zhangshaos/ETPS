#include <cstdio>
#include <chrono>
#include <ccs.h>
#include <print_cv_mat.h>
#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>


void parse(){
  //todo wrap main with argparse
}

const std::string
//img_path = "../demo/test_img/texture_compo.png";
//img_path = "../demo/test_img/UI_seq17_000700.jpg";
//img_path = "../demo/test_img/6h00002.jpg";
img_path = "../demo/test_img/UD_000261.jpg";


int main(int argc, char *argv[]) {
  auto img = cv::imread(img_path);
  printf("%s\n", cv::typeToString(img.type()).c_str());
  cv::Mat_<cv::Vec3b> rgb_img = img;
  printf("%s\n", cv::typeToString(rgb_img.type()).c_str());
  cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);
  cv::Mat lab_img;
  cv::cvtColor(rgb_img, lab_img, cv::COLOR_RGB2Lab);
  printf("%s\n", cv::typeToString(lab_img.type()).c_str());

  HyperParams params;
  params.verbose = false;
  namespace time = std::chrono;
  auto t0 = time::steady_clock::now();
  auto spx_label = ccs(rgb_img, params);
  auto t1 = time::steady_clock::now();
  float used_sec = time::duration_cast<time::milliseconds>(t1 - t0).count() * 1e-3f;
  printf("ccs algorithm cost %.3f second", used_sec);
  save_segmentation_map("seg.png", spx_label);
  save_image_with_segmentation("color_seg.png", rgb_img, spx_label);
  return 0;
}