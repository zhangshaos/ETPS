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
    "085110107700059",
    "085110107700068",
    "085110107700083",
    "085110107700146",
    "085110107700200",
    "085110107700240",
    "085110107700298",
    "085110107700345",
    "085110107700392",
    "085110107700551",
    "085110107700651",
    "085110107700695",
    "085110107700737",
    "085110107700807",
    "085110107700859",
    "085110107700865",
    "085110107701010",
    "085110107701050",
    "085110107701057",
    "085110107701119",
    "085110107701155",
    "085110107701208",
    "085110107701217",
    "085110107701225",
    "085110107701227",
    "085110107701272",
    "085110107701282",
    "085110107701308",
};


void
main_(int i){
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

  auto result_path1 = str_printf(512, "%s_spx_seg.png", names[i].c_str()),
       result_path2 = str_printf(512, "%s_spx_color.png", names[i].c_str());
  save_segmentation_map(result_path1, spx_label);
  save_image_with_segmentation(result_path2, rgb_img, spx_label);

  cv::Mat_<int>
      class_label = naive_semantic_seg(spx_label, rgb_img, class_prob);

  result_path1 = str_printf(512, "%s_naive_seg.png", names[i].c_str()),
  result_path2 = str_printf(512, "%s_naive_color.png", names[i].c_str());
  save_segmentation_map(result_path1, class_label);
  save_image_with_segmentation(result_path2, rgb_img, class_label);

  CRF_Params params2;
  params2.verbose = false;
  class_label = crf_semantic_seg(spx_label, rgb_img, class_prob, params2);

  result_path1 = str_printf(512, "%s_crf_seg.png", names[i].c_str());
  result_path2 = str_printf(512, "%s_crf_color.png", names[i].c_str());
  save_segmentation_map(result_path1, class_label);
  save_image_with_segmentation(result_path2, rgb_img, class_label);

  MRF_Params params3;
  params3.verbose = false;
  class_label = mrf_semantic_seg(spx_label, rgb_img, class_prob, params3);

  result_path1 = str_printf(512, "%s_mrf_seg.png", names[i].c_str());
  result_path2 = str_printf(512, "%s_mrf_color.png", names[i].c_str());
  save_segmentation_map(result_path1, class_label);
  save_image_with_segmentation(result_path2, rgb_img, class_label);
}


int main(int argc, char *argv[]) {
  printf("input the file index(0~%zu):\n"
         "the index out of range means take all files.\n",
         std::size(names)-1);
  fflush(stdout);
  int i=0;
  scanf("%d", &i);
  if (i < 0 || i > std::size(names))
    for (int j=std::size(names)-1; j>=0; --j)
      main_(j);
  else
    main_(i);
  return 0;
}