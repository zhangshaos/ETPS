#include <cstdio>
#include <chrono>
#include <etps.h>
#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>


void parse(){
  //todo wrap main with argparse
}

const std::string
//img_path = "../demo/test_img/texture_compo.png";
img_path = "../demo/test_img/UI_seq17_000700.jpg";
//img_path = "../demo/test_img/6h00002.jpg";
//img_path = "../demo/test_img/UD_000261.jpg";


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
  params.expect_spx_num = 500;
  params.spatial_scale = 0.1;

  namespace time = std::chrono;
  auto t0 = time::steady_clock::now();
  etps(rgb_img, params);
  auto t1 = time::steady_clock::now();
  float used_sec = time::duration_cast<time::milliseconds>(t1 - t0).count() * 1e-3f;
  printf("etps algorithm cost %.3f second", used_sec);
  return 0;
}