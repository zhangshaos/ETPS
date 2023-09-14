#include <cstdio>
#include <chrono>
#include <ccs.h>
#include <ccs_semantic_seg.h>
#include <npy2mat.h>
#include <print_cv_mat.h>
#include <opencv2/opencv.hpp>


const std::string
img_dir   = "/mnt/f/XGrids/PHOTO/0_resized",
label_dir = "/mnt/f/XGrids/PHOTO/0_result/pred",
names[]   = {
    "085110107700026",
    "085110107701308",
};



int main(int argc, char *argv[]) {
  printf("input the file index:\n");
  fflush(stdout);
  int i=0;
  scanf("%d", &i);
  if (i < 0 || i > std::size(names)){
    printf("file index must to be in the range [%d, %zu).\n", 0, std::size(names));
    return EXIT_SUCCESS;
  }

  auto img_path   = str_printf(512, "%s/%s.JPG", img_dir.c_str(), names[i].c_str()),
       label_path = str_printf(512, "%s/%s.npy", label_dir.c_str(), names[i].c_str());
  cv::Mat_<cv::Vec3b>
  rgb_img     = cv::imread(img_path);
  cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);
  cv::Mat_<float>
  class_prob  = cvDNN::blobFromNPY(label_path, CV_32F);

  HyperParams params1;
  params1.verbose = false;
  cv::Mat_<int>
  spx_label   = ccs(rgb_img, params1);

  cv::Mat_<int>
  class_label = naive_semantic_seg(spx_label, rgb_img, class_prob);

  auto result_path1 = str_printf(512, "%s_naive_seg.png", names[i].c_str()),
       result_path2 = str_printf(512, "%s_naive_color.png", names[i].c_str());
  save_segmentation_map(result_path1, class_label);
  save_image_with_segmentation(result_path2, rgb_img, class_label);

  CRF_Params params2;
  params2.wi = 10;
  class_label = crf_semantic_seg(spx_label, rgb_img, class_prob, params2);

  result_path1 = str_printf(512, "%s_crf_seg.png", names[i].c_str());
  result_path2 = str_printf(512, "%s_crf_color.png", names[i].c_str());
  save_segmentation_map(result_path1, class_label);
  save_image_with_segmentation(result_path2, rgb_img, class_label);

  MRF_Params params3;
  params3.wi = 10;
  class_label = mrf_semantic_seg(spx_label, rgb_img, class_prob, params3);

  result_path1 = str_printf(512, "%s_mrf_seg.png", names[i].c_str());
  result_path2 = str_printf(512, "%s_mrf_color.png", names[i].c_str());
  save_segmentation_map(result_path1, class_label);
  save_image_with_segmentation(result_path2, rgb_img, class_label);

//  HyperParams params;
//  params.expect_spx_num = 1000;
//  params.verbose = true;
//  namespace time = std::chrono;
//  auto t0 = time::steady_clock::now();
//  ccs(rgb_img, params);
//  auto t1 = time::steady_clock::now();
//  float used_sec = time::duration_cast<time::milliseconds>(t1 - t0).count() * 1e-3f;
//  printf("ccs algorithm cost %.3f second", used_sec);
  return 0;
}