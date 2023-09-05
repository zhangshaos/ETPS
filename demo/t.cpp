#include <cstdio>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>


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
  cv::Mat_<uchar> gray_img;
  cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);

  auto lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);
  std::vector<cv::Vec4i> lines;
  lsd->detect(gray_img, lines);
  auto vis_img = rgb_img.clone();
  lsd->drawSegments(vis_img, lines);
  cv::imwrite("lsd.png", vis_img);

  //经过测试LSD和EDPF旗鼓相当，由于LSD是直线，线段数量更少
  return 0;
}