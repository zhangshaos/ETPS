#include "ccs_semantic_seg.h"
#include "self_check_float.h"
#include "print_cv_mat.h"
#include "preprocess.h"
#include <algorithm>
#undef NDEBUG
#include <cassert>


inline
uchar
clamp2uchar(F64 v){
  return v > 0 ? (v < 255 ? (uchar) v : 255) : 0;
}


inline
SpxGraph
create_spx_graph_no_semantic(const cv::Mat_<int> &spx_label,
                             const cv::Mat_<cv::Vec3b> &rgb_img){
  constexpr int offset[][2] = {
      //{-1, -1},//dx,dy
      {0, -1},
      //{1, -1},
      {-1, 0},
      {1, 0},
      //{-1, 1},
      {0, 1},
      //{1, 1},
  };
  double max_v = 0;
  cv::minMaxIdx(spx_label, nullptr, &max_v);
  const int n_spx = max_v + 1;
  SpxGraph graph;
  graph.spx_s.resize(n_spx);
  graph.adjacent.resize(n_spx);

  cv::Mat_<uchar> extended(spx_label.rows, spx_label.cols, (uchar) 0);
  for (int y0 = 0; y0 < spx_label.rows; ++y0)
    for (int x0 = 0; x0 < spx_label.cols; ++x0) {
      if (extended.at<uchar>(y0, x0))
        continue;
      const int id0 = spx_label.at<int>(y0, x0);
      //DFS找到所有同ID的像素
      std::vector<cv::Point2i> accessed;
      std::vector<cv::Point2i> extend_stack;
      extend_stack.emplace_back(x0, y0);
      extended.at<uchar>(y0, x0) = 1;
      while (!extend_stack.empty()) {
        auto top = extend_stack.back();
        extend_stack.pop_back();
        accessed.emplace_back(top);
        assert(extended.at<uchar>(top.y, top.x));

        for (int j = std::size(offset) - 1; j >= 0; --j) {
          int y = top.y + offset[j][1],
              x = top.x + offset[j][0];
          if (y < 0 || y >= spx_label.rows ||
              x < 0 || x >= spx_label.cols)
            continue;
          const int id1 = spx_label.at<int>(y, x);
          if (id1 != id0) {
            graph.adjacent[id0].emplace(id1);
            graph.adjacent[id1].emplace(id0);
            continue;
          }
          if (extended.at<uchar>(y, x))
            continue;
          extend_stack.emplace_back(x, y);
          extended.at<uchar>(y, x) = 1;
        }
      }
      auto &spx0 = graph.spx_s[id0];
      F64 x_sum = 0, y_sum = 0,
          r_sum = 0, g_sum = 0, b_sum = 0;
      for (auto p : accessed) {
        auto rgb = rgb_img.at<cv::Vec3b>(p.y, p.x);
        x_sum += (F64) p.x;
        y_sum += (F64) p.y;
        r_sum += (F64) rgb[0];
        g_sum += (F64) rgb[1];
        b_sum += (F64) rgb[2];
      }
      if (spx0.is_valid){
        //label中可能存在空间上不相邻，但是ID一致的多个簇
        assert(!spx0.pxs.empty());
        x_sum += (F64) spx0.xy_mean.x * (F64) spx0.pxs.size();
        y_sum += (F64) spx0.xy_mean.y * (F64) spx0.pxs.size();
        r_sum += (F64) spx0.rgb_mean[0] * (F64) spx0.pxs.size();
        g_sum += (F64) spx0.rgb_mean[1] * (F64) spx0.pxs.size();
        b_sum += (F64) spx0.rgb_mean[2] * (F64) spx0.pxs.size();
        accessed.insert(accessed.end(), spx0.pxs.begin(), spx0.pxs.end());
      }
      spx0.spx_id = id0;
      spx0.semantic_logits;//暂不更新语义ID
      spx0.is_valid = true;
      spx0.xy_mean = {
          int(x_sum / (F64) accessed.size()),
          int(y_sum / (F64) accessed.size()),
      };
      spx0.rgb_mean = {
          clamp2uchar(r_sum / (F64) accessed.size()),
          clamp2uchar(g_sum / (F64) accessed.size()),
          clamp2uchar(b_sum / (F64) accessed.size()),
      };
      spx0.pxs = std::move(accessed);
    }

#ifndef NDEBUG
  int count=0;
  for (const auto spx : graph.spx_s)
    count += spx.pxs.size();
  assert(count == spx_label.rows * spx_label.cols);
#endif
  return graph;
}


SpxGraph
create_spx_graph(const cv::Mat_<int> &spx_label,
                 const cv::Mat_<cv::Vec3b> &rgb_img,
                 const cv::Mat_<float> &soft_semantic_score){
  assert(soft_semantic_score.size.dims() == 3 &&
         "soft semantic label must to be shape of (H,W,C).");
  const int H = soft_semantic_score.size[0],
            W = soft_semantic_score.size[1],
            C = soft_semantic_score.size[2];
  auto graph = create_spx_graph_no_semantic(spx_label, rgb_img);
  for (auto &spx : graph.spx_s){
    if (!spx.is_valid)
      continue;
    std::vector<F64> score(C, (F64)0);
    for (auto px : spx.pxs)
      for (int i=0; i<C; ++i) {
        float s = soft_semantic_score.at<float>(px.y, px.x, i);
        score[i] += (F64)s;
      }
    spx.semantic_logits.resize(C, 0.f);
    for (int j=C-1; j>=0; --j)
      spx.semantic_logits[j] = (score[j] / (F64)spx.pxs.size());
  }
  return graph;
}

cv::Mat_<int>
arg_max(const cv::Mat_<float> &mat){
  CV_Assert(mat.dims == 3);
  const int H=mat.size[0], W=mat.size[1], C=mat.size[2];
  cv::Mat_<int> result;
  result.create(H, W);
  for (int y=0; y<H; ++y)
    for (int x=0; x<W; ++x){
      float max_v   = 0;
      int arg_max_v = -1;
      for (int c=0; c<C; ++c){
        float v = mat.at<float>(y, x, c);
        if (v <= max_v)
          continue;
        max_v     = v;
        arg_max_v = c;
      }
      result.at<int>(y, x) = arg_max_v;
    }
  return result;
}


cv::Mat_<int>
naive_semantic_seg(const cv::Mat_<int> &spx_label,
                   const cv::Mat_<cv::Vec3b> &rgb_img_,
                   const cv::Mat_<float> &soft_semantic_score){
  auto rgb_img = rgb2lab(rgb_img_, true);
  auto graph  = create_spx_graph(spx_label, rgb_img, soft_semantic_score);
  cv::Mat_<int> result(spx_label.rows, spx_label.cols, -1);
  for (const auto &spx : graph.spx_s){
    if (!spx.is_valid)
      continue;
    const auto &score = spx.semantic_logits;
    int max_class = std::distance(score.begin(), std::max_element(score.begin(), score.end()));
    assert(max_class >= 0);
    for (auto px : spx.pxs)
      result.at<int>(px.y, px.x) = max_class;
  }
#ifndef NDEBUG
  double min_v=0, max_v=0;
  cv::minMaxIdx(result, &min_v, &max_v);
  assert(min_v >= 0 && "logic error! SpxGraph do not contain all pixel!");
#endif
  return result;
}


#pragma GCC optimize("O0")

//使用条件随机场CDF优化语义图（去噪）
//每个超像素的语义作为随机变量Yi，当前的语义图作为当前观察值X，两个相邻的超像素之间存在一条边。
//使用ICM迭代条件模式优化，寻找局部最大值P(Y|X)
inline
std::vector<int>
crf_init_variable(const SpxGraph &graph){
  std::vector<int> variable(graph.spx_s.size(), -1);
  for (int j=graph.spx_s.size()-1; j>=0; --j){
    const auto &spx = graph.spx_s[j];
    if (!spx.is_valid)
      continue;
    const auto &score = spx.semantic_logits;
    int max_class = std::distance(score.begin(),
                                  std::max_element(score.begin(), score.end()));
    assert(max_class >= 0);
    variable[j] = max_class;
  }
  return variable;
}

inline
F32
energy(int i, int yi,
       const SpxGraph &graph){
  F32 e = graph.spx_s[i].semantic_logits[yi];
  return e;
}

inline
F32
energy(int i, int yi, int j, int yj,
       const SpxGraph &graph,
       const cv::Vec3b yi_mean_rgb){
  const F32 max_d_rgb = 25.f * 25.f;
  F32 e = 0;
  if (yi == yj){
    F32 dr = graph.spx_s[i].rgb_mean[0] - yi_mean_rgb[0],
        dg = graph.spx_s[i].rgb_mean[1] - yi_mean_rgb[1],
        db = graph.spx_s[i].rgb_mean[2] - yi_mean_rgb[2];
    F32 d = (dr * dr + dg * dg + db * db) / max_d_rgb;
    e = (F32)1 - d;
  } else {
    F32 dr = graph.spx_s[i].rgb_mean[0] - graph.spx_s[j].rgb_mean[0],
        dg = graph.spx_s[i].rgb_mean[1] - graph.spx_s[j].rgb_mean[1],
        db = graph.spx_s[i].rgb_mean[2] - graph.spx_s[j].rgb_mean[2];
    e = (dr * dr + dg * dg + db * db) / max_d_rgb;
  }
  return e;
}

inline
F32
log_conditional_prob(int i,
                     int yi,
                     const std::vector<int> &variable,
                     const SpxGraph &graph,
                     const std::vector<cv::Vec3b> &class_mean_rgb,
                     const CRF_Params &params){
  assert(variable.size() == graph.spx_s.size());
  assert(graph.spx_s[i].is_valid);
  //求解条件概率P(yi|Y\yi,X)，yi和Y-{yi和yi附近变量}在已知yi附近随机变量取值后条件独立
  F32 prob = params.wi * energy(i, yi, graph);
  if (params.verbose && i == variable.size()/2)
    printf("%s(i=%d,yi=%d): prob=%f\n", __func__, i, yi, (float)prob);
  for (int j : graph.adjacent[i]){
    assert(graph.spx_s[j].is_valid);
    prob += params.wj * energy(i, yi, j, variable[j], graph, class_mean_rgb[i]);
    if (params.verbose && i == variable.size()/2)
      printf("prob=%f\n", (float)prob);
  }
  return prob;
}

std::vector<cv::Vec3b>
mean_class_rgb(const SpxGraph &graph,
               const std::vector<int> &variable,
               const int num_class){
  struct RGB_Count{
    F64 r=0, g=0, b=0;
    int count = 0;
  };
  std::vector<RGB_Count> mean_rgb;
  mean_rgb.resize(num_class);
  assert(variable.size() == graph.spx_s.size());
  for (int j=graph.spx_s.size()-1; j>=0; --j){
    const auto &spx = graph.spx_s[j];
    if (!spx.is_valid)
      continue;
    auto &rgb = mean_rgb[variable[j]];
    const F64 n = spx.pxs.size();
    rgb.r += spx.rgb_mean[0] * n;
    rgb.g += spx.rgb_mean[1] * n;
    rgb.b += spx.rgb_mean[2] * n;
    rgb.count += spx.pxs.size();
  }
  std::vector<cv::Vec3b> result(num_class);
  for (int j=num_class-1; j>=0; --j) {
    const F64 n = mean_rgb[j].count;
    if (n > 0) {
      uchar r = clamp2uchar(mean_rgb[j].r / n),
          g = clamp2uchar(mean_rgb[j].g / n),
          b = clamp2uchar(mean_rgb[j].b / n);
      result[j] = cv::Vec3b{r, g, b};
    } else {
      result[j] = cv::Vec3b::zeros();
    }
  }
  return result;
}

inline
cv::Mat_<int>
graph2label(const SpxGraph &spx_graph,
            const std::vector<int> &spx_label,
            int h, int w){
  assert(spx_graph.spx_s.size() == spx_label.size());
  cv::Mat_<int> result(h, w, -1);
  for (int j=spx_label.size()-1; j>=0; --j){
    const auto &spx = spx_graph.spx_s[j];
    if (!spx.is_valid)
      continue;
    int c = spx_label[j];
    for (auto px : spx.pxs)
      result.at<int>(px.y, px.x) = c;
  }
  return result;
}

cv::Mat_<int>
crf_semantic_seg(const cv::Mat_<int> &spx_label,
                 const cv::Mat_<cv::Vec3b> &rgb_img_,
                 const cv::Mat_<float> &soft_semantic_score,
                 const CRF_Params &params){
  auto rgb_img = rgb2lab(rgb_img_, true);
  auto graph = create_spx_graph(spx_label, rgb_img, soft_semantic_score);
  const int H = soft_semantic_score.size[0],
            W = soft_semantic_score.size[1],
            C = soft_semantic_score.size[2];
  auto Y = crf_init_variable(graph);
  for (int iter=0; iter<params.max_iter_num; ++iter){
    if (params.verbose) {
      printf("start iteration epoch %d...\n", iter);
      auto mat = graph2label(graph, Y, H, W);
      save_segmentation_map(
          str_printf(512, "%s_seg_%d.png", __func__, iter), mat);
      save_image_with_segmentation(
          str_printf(512, "%s_color_%d.png", __func__, iter), rgb_img_, mat);
    }

    auto class_mean_rgb = mean_class_rgb(graph, Y, C);
    int i       = (iter % 2 == 0) ? 0 : graph.spx_s.size()-1,
        i_end   = (iter % 2 == 0) ? graph.spx_s.size()-1 : 0,
        step    = (iter % 2 == 0) ? 1 : -1;
    bool changed = false;
    while (i <= i_end){
      if (graph.spx_s[i].is_valid){
        //求解条件概率P(yi|Y\yi,X)，找到使其最大的yi
        int max_class = -1;
        F32 max_class_prob = std::numeric_limits<float>::min();
        for (int c=0; c < C; ++c){
          F32 prob = log_conditional_prob(i, c, Y, graph,
                                          class_mean_rgb,
                                          params);
          if (prob > max_class_prob){
            max_class_prob = prob;
            max_class      = c;
          }
        }
        assert(max_class >= 0);
        if (Y[i] != max_class)
          changed = true;
        Y[i] = max_class;
      }
      i += step;
    }
    if (!changed)
      break;
  }

  auto result = graph2label(graph, Y, H, W);
#ifndef NDEBUG
  double min_v=0;
  cv::minMaxIdx(result, &min_v);
  assert(min_v >= 0 && "logic error! SpxGraph do not contain all pixel!");
#endif
  return result;
}
