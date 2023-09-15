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

inline
std::vector<cv::Vec3b>
class_rgb_mean(const SpxGraph &graph,
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
      result[j] = cv::Vec3b{0, 0, 0};
    }
  }
  return result;
}


//算法一：使用条件随机场CRF优化语义图（去噪）
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
crf_energy(int i, int yi,
           const SpxGraph &graph){
  F32 e = graph.spx_s[i].semantic_logits[yi];
  return e;
}

inline
F32
crf_energy(int i, int yi,
           int j, int yj,
           const SpxGraph &graph){
  const F32 max_d_rgb = 50.f * 50.f;
  F32 ri = graph.spx_s[i].rgb_mean[0],
      rj = graph.spx_s[j].rgb_mean[0],
      gi = graph.spx_s[i].rgb_mean[1],
      gj = graph.spx_s[j].rgb_mean[1],
      bi = graph.spx_s[i].rgb_mean[2],
      bj = graph.spx_s[j].rgb_mean[2];
  F32 dr = ri - rj,
      dg = gi - gj,
      db = bi - bj;
  F32 e = (dr * dr + dg * dg + db * db) / max_d_rgb;
  //倾向于对颜色差别较大的赋予不同的类别
  if (yi == yj)
    e *= (F32) 0.01f;
  else
    e *= (F32) 1;
  return e;
}

inline
F32
crf_log_conditional_prob(int i, int yi,
                         const std::vector<int> &variable,
                         const SpxGraph &graph,
                         const CRF_Params &params){
  assert(variable.size() == graph.spx_s.size());
  assert(graph.spx_s[i].is_valid);
  //求解条件概率P(yi|Y\yi,X)，yi和Y-{yi和yi附近变量}在已知yi附近随机变量取值后条件独立
  F32 prob = params.wi * crf_energy(i, yi, graph);
  if (params.verbose && i == variable.size()/2)
    printf("%s(i=%d,yi=%d): prob=%f\n", __func__, i, yi, (float)prob);
  for (int j : graph.adjacent[i]){
    assert(graph.spx_s[j].is_valid);
    prob += params.wj * crf_energy(i, yi, j, variable[j], graph);
    if (params.verbose && i == variable.size()/2)
      printf("prob=%f\n", (float)prob);
  }
  return prob;
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

    auto newY = Y;
    int i       = (iter % 2 == 0) ? 0 : graph.spx_s.size()-1,
        i_end   = (iter % 2 == 0) ? graph.spx_s.size()-1 : 0,
        step    = (iter % 2 == 0) ? 1 : -1;
    bool changed = false;
    while (i <= i_end){//todo parallel
      if (!graph.spx_s[i].is_valid){
        i += step;
        continue;
      }
      //求解条件概率P(yi|Y\yi,X)，找到使其最大的yi
      int max_class = -1;
      F32 max_class_prob = -std::numeric_limits<float>::max();
      for (int c=0; c < C; ++c){
        F32 prob = crf_log_conditional_prob(i, c, Y, graph, params);
        if (prob > max_class_prob){
          max_class_prob = prob;
          max_class      = c;
        }
      }
      assert(max_class >= 0);
      if (Y[i] != max_class)
        changed = true;
      newY[i] = max_class;
      i += step;
    }

    if (!changed)
      break;
    Y.swap(newY);
  }

  auto result = graph2label(graph, Y, H, W);
#ifndef NDEBUG
  double min_v=0;
  cv::minMaxIdx(result, &min_v);
  assert(min_v >= 0 && "logic error! SpxGraph do not contain all pixel!");
#endif
  return result;
}


//算法二：使用马尔科夫随机场优化语义图（语义分割）
//每个超像素的语义作为随机变量Yi，当前的像素图作为当前观察值X，两个相邻的超像素之间存在一条边。
//使用ICM迭代条件模式优化，寻找局部最大值P(Y|X) = P(Y) * P(X|Y) / P(X)


inline
std::vector<int>
mrf_init_variable(const SpxGraph &graph){
  return crf_init_variable(graph);
}

struct Class2RGB_Distribution{
  struct RGB_NormalDist{
    F32 mu_r = (F32)0,
        mu_g = (F32)0,
        mu_b = (F32)0,
        sigma_r = (F32)0,
        sigma_g = (F32)0,
        sigma_b = (F32)0;
    bool valid  = false;
    F32
    log_prob(cv::Vec3b rgb)const{
      assert(valid);
      F32 r = rgb[0] - mu_r,
          g = rgb[1] - mu_g,
          b = rgb[2] - mu_b;
      F32 log_p0 = /*-(log_sqrt_2pi + std::log(sigma_r))*/ - 0.5f * (r * r) / (sigma_r * sigma_r),
          log_p1 = /*-(log_sqrt_2pi + std::log(sigma_g))*/ - 0.5f * (g * g) / (sigma_g * sigma_g),
          log_p2 = /*-(log_sqrt_2pi + std::log(sigma_b))*/ - 0.5f * (b * b) / (sigma_b * sigma_b);
      F32 log_p  = log_p0 + log_p1 + log_p2;
      return log_p;
    }

    static inline
    F32 log_sqrt_2pi = (0.5f * std::log(2.f * (float)M_PI));
  };

  int class_num=0;
  std::vector<RGB_NormalDist> dist_s;
};

inline
Class2RGB_Distribution
mrf_create_crgb_distribution(const cv::Mat_<cv::Vec3b> &rgb_img,
                             const SpxGraph &graph,
                             const std::vector<int> &variable,
                             const int num_class){
  assert(variable.size() == graph.spx_s.size());

  struct RGB_Count {
    F64 r=(F64)0, g=(F64)0, b=(F64)0;
    int count=0;
  };
  std::vector<RGB_Count> mean_rgb_count(num_class);
  for (int j=variable.size()-1; j>=0; --j){
    int c = variable[j];
    const auto &spx = graph.spx_s[j];
    if (!spx.is_valid)
      continue;
    const int n = spx.pxs.size();
    mean_rgb_count[c].count  += n;
    mean_rgb_count[c].r      += (F64)spx.rgb_mean[0] * (F64)n;
    mean_rgb_count[c].g      += (F64)spx.rgb_mean[1] * (F64)n;
    mean_rgb_count[c].b      += (F64)spx.rgb_mean[2] * (F64)n;
  }
  for (int c=num_class-1; c>=0; --c){
    if (mean_rgb_count[c].count <= 0)
      continue;
    const F64 n = mean_rgb_count[c].count;
    mean_rgb_count[c].r /= n;
    mean_rgb_count[c].g /= n;
    mean_rgb_count[c].b /= n;
  }

  std::vector<RGB_Count> var_rgb_count(num_class);
  for (int j=variable.size()-1; j>=0; --j){
    int c = variable[j];
    const auto &spx = graph.spx_s[j];
    if (!spx.is_valid || mean_rgb_count[c].count <= 0)
      continue;
    var_rgb_count[c].count += spx.pxs.size();
    F32 r_mu = mean_rgb_count[c].r,
        g_mu = mean_rgb_count[c].g,
        b_mu = mean_rgb_count[c].b;
    for (auto px : spx.pxs){
      auto rgb = rgb_img.at<cv::Vec3b>(px.y, px.x);
      F32 dr = r_mu - (F32)rgb[0],
          dg = g_mu - (F32)rgb[1],
          db = b_mu - (F32)rgb[2];
      var_rgb_count[c].r += dr*dr;
      var_rgb_count[c].g += dg*dg;
      var_rgb_count[c].b += db*db;
    }
  }
  for (int c=num_class-1; c>=0; --c){
    if (var_rgb_count[c].count <= 0)
      continue;
    const F64 n = var_rgb_count[c].count;
    var_rgb_count[c].r = std::sqrt(var_rgb_count[c].r / n);
    var_rgb_count[c].g = std::sqrt(var_rgb_count[c].g / n);
    var_rgb_count[c].b = std::sqrt(var_rgb_count[c].b / n);
  }

  Class2RGB_Distribution dist_s;
  dist_s.class_num = num_class;
  dist_s.dist_s.resize(num_class);
  for (int c=num_class-1; c>=0; --c){
    const int n = mean_rgb_count[c].count;
    assert(n == var_rgb_count[c].count);
    if (n <= 0)
      continue;
    auto &dist = dist_s.dist_s[c];
    dist.mu_r     = mean_rgb_count[c].r;
    dist.mu_g     = mean_rgb_count[c].g;
    dist.mu_b     = mean_rgb_count[c].b;
    dist.sigma_r  = var_rgb_count[c].r;
    dist.sigma_g  = var_rgb_count[c].g;
    dist.sigma_b  = var_rgb_count[c].b;
    dist.valid    = true;
  }
  return dist_s;
}

//求解条件概率P(X|Y)，假设超像素颜色之间彼此独立=>P(xi|Y)，假设超像素颜色只与yi有关=>P(xi|yi)
inline
F32
mrf_log_likelihood(int yi,
               cv::Vec3b rgb_i,
               const Class2RGB_Distribution &class2rgb_dist){
  assert(yi >= 0 && yi < class2rgb_dist.class_num);
  const auto &dist = class2rgb_dist.dist_s[yi];
  if (!dist.valid)
    return (F32)-1e3f;
  F32 prob = dist.log_prob(rgb_i);
  return prob;
}

inline
F32
mrf_energy(int i, int yi,
           const SpxGraph &graph){
  return crf_energy(i, yi, graph);
}

inline
F32
mrf_energy(int i, int yi,
           int j, int yj,
           const SpxGraph &graph){
  return crf_energy(i, yi, j, yj, graph);
}

inline
F32
mrf_log_prob(int i, int yi,
             const std::vector<int> &variable,
             const SpxGraph &graph,
             const MRF_Params &params){
  assert(variable.size() == graph.spx_s.size());
  assert(graph.spx_s[i].is_valid);
  //求解概率P(yi)
  F32 prob = params.wi * mrf_energy(i, yi, graph);
  if (params.verbose && i == variable.size()/2)
    printf("%s(i=%d,yi=%d): prob=%f\n", __func__, i, yi, (float)prob);
  for (int j : graph.adjacent[i]){
    assert(graph.spx_s[j].is_valid);
    prob += params.wj * mrf_energy(i, yi, j, variable[j], graph);
    if (params.verbose && i == variable.size()/2)
      printf("prob=%f\n", (float)prob);
  }
  if (prob <= 0)
    prob = (F32)1e-3;
  return std::log(prob);
}

cv::Mat_<int>
mrf_semantic_seg(const cv::Mat_<int> &spx_label,
                 const cv::Mat_<cv::Vec3b> &rgb_img_,
                 const cv::Mat_<float> &soft_semantic_score,
                 const MRF_Params &params){
  auto rgb_img = rgb2lab(rgb_img_, true);
  auto graph = create_spx_graph(spx_label, rgb_img, soft_semantic_score);
  const int H = soft_semantic_score.size[0],
            W = soft_semantic_score.size[1],
            C = soft_semantic_score.size[2];
  auto Y = mrf_init_variable(graph);
  for (int iter=0; iter<params.max_iter_num; ++iter){
    if (params.verbose) {
      printf("start iteration epoch %d...\n", iter);
      auto mat = graph2label(graph, Y, H, W);
      save_segmentation_map(
          str_printf(512, "%s_seg_%d.png", __func__, iter), mat);
      save_image_with_segmentation(
          str_printf(512, "%s_color_%d.png", __func__, iter), rgb_img_, mat);
    }

    auto crgb_distribution = mrf_create_crgb_distribution(rgb_img, graph, Y, C);
    auto newY = Y;
    int i       = (iter % 2 == 0) ? 0 : graph.spx_s.size()-1,
        i_end   = (iter % 2 == 0) ? graph.spx_s.size()-1 : 0,
        step    = (iter % 2 == 0) ? 1 : -1;
    bool changed = false;
    while (i <= i_end){//todo parallel
      if (!graph.spx_s[i].is_valid){
        i += step;
        continue;
      }
      //比较条件概率P(yi|Y\yi,X)，找到使其最大的yi
      int max_class = -1;
      F32 max_class_prob = -std::numeric_limits<float>::max();
      for (int c=0; c < C; ++c){
        F32 prob_yi    = mrf_log_prob(i, c, Y, graph, params);
        F32 likelihood = mrf_log_likelihood(c, graph.spx_s[i].rgb_mean, crgb_distribution);
        F32 prob       = prob_yi + likelihood * (F32)0.1f;
        if (prob > max_class_prob){
          max_class_prob = prob;
          max_class      = c;
        }
      }
      assert(max_class >= 0);
      if (Y[i] != max_class)
        changed = true;
      newY[i] = max_class;
      i += step;
    }

    if (!changed)
      break;
    Y.swap(newY);
  }

  auto result = graph2label(graph, Y, H, W);
#ifndef NDEBUG
  double min_v=0;
  cv::minMaxIdx(result, &min_v);
  assert(min_v >= 0 && "logic error! SpxGraph do not contain all pixel!");
#endif
  return result;
}