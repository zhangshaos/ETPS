#ifndef CCS_SRC_CCS_SEMANTIC_SEG_H_
#define CCS_SRC_CCS_SEMANTIC_SEG_H_


#include <opencv2/core.hpp>
#include <set>
#include <vector>


struct Spx {
  std::vector<cv::Point2i> pxs;
  std::vector<float> semantic_logits;
  int spx_id{ -1 };
  cv::Point2i xy_mean{ -1, -1 };
  cv::Vec3b rgb_mean{ 0, 0, 0 };
  bool is_valid{ false };
};


struct SpxGraph {
  std::vector<Spx> spx_s;
  std::vector<std::set<int> > adjacent;
};


SpxGraph
create_spx_graph(const cv::Mat_<int> &spx_label,
                 const cv::Mat_<cv::Vec3b> &rgb_img,
                 const cv::Mat_<float> &soft_semantic_score);


cv::Mat_<int>
naive_semantic_seg(const cv::Mat_<int> &spx_label,
                   const cv::Mat_<cv::Vec3b> &rgb_img,
                   const cv::Mat_<float> &soft_semantic_score);


struct CRF_Params{
  int max_iter_num=10;
  float wi=10, wj=1;//todo config it
  bool verbose=true;
};

cv::Mat_<int>
crf_semantic_seg(const cv::Mat_<int> &spx_label,
                 const cv::Mat_<cv::Vec3b> &rgb_img,
                 const cv::Mat_<float> &soft_semantic_score,
                 const CRF_Params &params);


#endif //CCS_SRC_CCS_SEMANTIC_SEG_H_
