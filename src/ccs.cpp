#include "ccs.h"
#include "self_check_float.h"
#include "print_cv_mat.h"
#include "preprocess.h"
#include "throw_exception.h"
#include "connectivity.h"
#include "geodesic_distance.h"
#include "disjoint_set.h"
#include <set>
#undef NDEBUG
#include <cassert>


inline
uchar
clamp2uchar(F64 v){
  return v > 0 ? (v < 255 ? (uchar) v : 255) : 0;
}

struct SuperPixel{
  const int id = -1;
  cv::Point2i xy_mean{ -1, -1 };
  cv::Vec3b rgb_mean{ 0, 0, 0 };
  int n_px = 0;
};

inline
F32
dist_rgb(const cv::Vec3b &l, const cv::Vec3b &r){
  int dr = (int) l[0] - (int) r[0],
      dg = (int) l[1] - (int) r[1],
      db = (int) l[2] - (int) r[2];
  F32 dist = (dr * dr + dg * dg + db * db);
  return dist;
}

inline
F32
dist_spatial(const cv::Point2i &l, const cv::Point2i &r){
  auto d = l - r;
  F32 dist = (F32) d.y * (F32) d.y + (F32) d.x * (F32) d.x;
  return dist;
}

void
initialize_hyper_params(const HyperParams &params, int diameter){
  F32 max_d_rgb              = 25.f * 25.f;
  F32 max_d_spatial          = 2.f * diameter * 0.5f * diameter * 0.5f;
  params.spatial_scale      *= ((float)max_d_rgb / (float)max_d_spatial);
  params.min_edge_threshold  = (0.5f * diameter);
  params.max_iter_num        = diameter + 1;
  params.expect_spx_diameter = diameter;
  printf("%s(): "
         "spatial_scale=%f, min_edge_threshold=%d, max_iter_num=%d, expect_spx_diameter=%d.\n",
         __func__,
         params.spatial_scale, params.min_edge_threshold,
         params.max_iter_num, params.expect_spx_diameter);
}


std::vector<cv::Point2i>
find_boundary_px(const cv::Mat_<int> &label,
                 const cv::Mat_<uchar> &edge,
                 bool exclude_edge) {
  constexpr int offset[][2] = {
      {-1, -1},// dx, dy
      { 0, -1},
      { 1, -1},
      {-1,  0},
      { 1,  0},
      {-1,  1},
      { 0,  1},
      { 1,  1}
  };
  std::vector<cv::Point2i> result;
  result.reserve(label.rows + label.cols);
  for(int y0=0; y0<label.rows; ++y0)
    for(int x0=0; x0<label.cols; ++x0){
      if (exclude_edge && edge.at<uchar>(y0, x0))
        //处于纹理边缘的像素不再优化，防止边缘不对齐
        continue;
      bool boundary = false;
      int id0 = label.at<int>(y0, x0);
      for (int j=std::size(offset)-1; j>=0; --j){
        int y = y0 + offset[j][1],
            x = x0 + offset[j][0];
        if (y < 0 || y >= label.rows || x < 0 || x >= label.cols)
          continue;
        int id = label.at<int>(y, x);
        if (id != id0){
          boundary = true;
          break;
        }
      }
      if (boundary)
        result.emplace_back(x0, y0);
    }
  return result;
}


std::vector<int>
adjacent_spx(cv::Point2i px,
             const cv::Mat_<int> &label) {
  std::vector<int> adj_spx;
  constexpr int offset[][2] = {
      {-1, -1},// dx, dy
      { 0, -1},
      { 1, -1},
      {-1,  0},
      { 1,  0},
      {-1,  1},
      { 0,  1},
      { 1,  1}
  };
  adj_spx.reserve(std::size(offset));
  for (int t=std::size(offset); t>=0; --t){
    int y = px.y + offset[t][1],
        x = px.x + offset[t][0];
    if (y < 0 || y >= label.rows || x < 0 || x >= label.cols)
      continue;
    int id = label.at<int>(y, x);
    adj_spx.emplace_back(id);
  }
  std::sort(adj_spx.begin(), adj_spx.end());
  adj_spx.erase(std::unique(adj_spx.begin(), adj_spx.end()), adj_spx.end());
  return adj_spx;
}


F32
dist_px2spx(cv::Point2i px,
            const SuperPixel &spx,
            const cv::Mat_<cv::Vec3b> &rgb_img,
            const cv::Mat_<uchar> &grad,
            const HyperParams &params) {
  auto px_rgb   = rgb_img.at<cv::Vec3b>(px.y, px.x);
  F32 d_rgb     = dist_rgb(px_rgb, spx.rgb_mean);
  F32 d_spatial = dist_spatial(px, spx.xy_mean);
  //F32 d_spatial = dist_spatial_geodesic(px, spx.xy_mean, grad);
  F32 d         = d_rgb + d_spatial * (F32)params.spatial_scale;
  while (params.verbose){
    static int r = 0;
    r = ++r % (params.max_iter_num * params.expect_spx_num);
    if (r > 0)
      //避免太多日志内容
      break;
    printf("px(%d,%d) to spx(%d, %d)(id=%d)\n"
           "\td(%f) = d_rgb(%f) + d_spatial(%f) * params.spatial_scale(%f)\n",
           px.x, px.y, spx.xy_mean.x, spx.xy_mean.y, spx.id,
           (float)d, (float)d_rgb, (float)d_spatial, (float)params.spatial_scale);
    break;
  }
  return d;
}


void
update_spx_from_label(std::vector<SuperPixel> &spx_s,
                      const cv::Mat_<int> &label,
                      const cv::Mat_<cv::Vec3b> &rgb_img){
  double max_v=0;
  cv::minMaxIdx(label, nullptr, &max_v);
  const int n_spx = max_v + 1;
  spx_s.resize(n_spx);

  struct RGB_XY_Count{
    cv::Vec<F64, 3> rgb{ 0, 0, 0 };
    cv::Vec<F64, 2> xy{ 0, 0 };
    int count{ 0 };
  };
  std::vector<RGB_XY_Count> px_counter;
  px_counter.resize(n_spx);

  for(int y=0; y<rgb_img.rows; ++y)
    for(int x=0; x<rgb_img.cols; ++x){
      int id      = label.at<int>(y, x);
      auto px_rgb = rgb_img.at<cv::Vec3b>(y, x);
      auto &xy    = px_counter[id].xy;
      auto &rgb   = px_counter[id].rgb;
      xy[0] += x;
      xy[1] += y;
      rgb[0] += px_rgb[0];
      rgb[1] += px_rgb[1];
      rgb[2] += px_rgb[2];
      px_counter[id].count++;
    }

  for (int j=spx_s.size()-1; j>=0; --j){
    int c = px_counter[j].count;
    if (c <= 0){
      spx_s[j].n_px = c;
      continue;
    }
    spx_s[j].n_px     = c;
    auto rgb_mean     = px_counter[j].rgb;
    spx_s[j].rgb_mean = {
        clamp2uchar(rgb_mean[0] / (F64) c),
        clamp2uchar(rgb_mean[1] / (F64) c),
        clamp2uchar(rgb_mean[2] / (F64) c)
    };
    auto xy_mean      = px_counter[j].xy;
    spx_s[j].xy_mean  = {
        (int) (xy_mean[0] / (F64) c),
        (int) (xy_mean[1] / (F64) c)
    };
  }
}


void
merge_small_spx(cv::Mat_<int> &label,
                const cv::Mat_<cv::Vec3b> &rgb_img,
                int num_max_merged_px,
                bool verbose) {
  constexpr int offset[][2] = {
      {-1, -1},//dx,dy
      {0, -1},
      {1, -1},
      {-1, 0},
      {1, 0},
      {-1, 1},
      {0, 1},
      {1, 1},
  };
  struct Spx {
    std::vector<cv::Point2i> pxs;
    int id{-1};
    cv::Point2i xy_mean{-1, -1};
    cv::Vec3b rgb_mean{0, 0, 0};
    bool is_valid{false};
  };
  struct SpxGraph {
    std::vector<Spx> spx_s;
    std::vector<std::set<int> > adjacent;
  };

  double max_v = 0;
  cv::minMaxIdx(label, nullptr, &max_v);
  const int n_spx = max_v + 1;
  SpxGraph graph;
  graph.spx_s.resize(n_spx);
  graph.adjacent.resize(n_spx);

  cv::Mat_<uchar> extended(label.rows, label.cols, (uchar) 0);
  for (int y0 = 0; y0 < label.rows; ++y0)
    for (int x0 = 0; x0 < label.cols; ++x0) {
      if (extended.at<uchar>(y0, x0))
        continue;
      const int id0 = label.at<int>(y0, x0);
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
          if (y < 0 || y >= label.rows ||
              x < 0 || x >= label.cols)
            continue;
          const int id1 = label.at<int>(y, x);
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
      spx0.id = id0;
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

  DisjointSet merge_set;
  std::vector<int> id_count(n_spx, 0);
  int count = 0;
  for (int j = graph.spx_s.size() - 1; j >= 0; --j){
    const auto &v = graph.spx_s[j];
    if (!v.is_valid)
      continue;
    assert(v.id == j);
    id_count[j] += v.pxs.size();
    count += v.pxs.size();
    const auto &e = graph.adjacent[j];
    assert(!e.count(j));//不能重复合并自己
    merge_set.try_add_class(j);
  }
  assert(count == label.rows * label.cols);

  //每个小块与周围最相似的块合并
  for (int j=graph.spx_s.size()-1; j>=0; --j){
    const auto &v0 = graph.spx_s[j];
    if (!v0.is_valid)
      continue;
    if (id_count[merge_set.find_root_class(j)] > num_max_merged_px)
      continue;
    auto &e0 = graph.adjacent[j];
    int most_similar_id = -1;
    F32 most_similar_dist = std::numeric_limits<float>::max();
    for (int i : e0){
      if (merge_set.find_root_class(j) == merge_set.find_root_class(i))
        //避免出现：两个小集合彼此最近，总数量小于阈值，但它们只能与彼此合并（距离最近）
        continue;
      const auto &v1 = graph.spx_s[i];
      F32 d = dist_rgb(v0.rgb_mean, v1.rgb_mean);
      if (d < most_similar_dist){
        most_similar_dist = d;
        most_similar_id   = i;
      }
    }
    assert(most_similar_id >= 0 && most_similar_id != j);
    const int n0 = id_count[merge_set.find_root_class(j)],
              n1 = id_count[merge_set.find_root_class(most_similar_id)];
    merge_set.union_class(j, most_similar_id);
    id_count[merge_set.find_root_class(j)] = n0 + n1;
  }

  //经过合并后，有效的ID值可能在数轴上分布太分散
  std::vector<uint32_t> shrink_id_map;
  int n_new_spx = merge_set.shrink_parent(&shrink_id_map);
  if (verbose)
    printf("%s(num_max_merged_px=%d): from %d spx to %d.\n",
           __func__, num_max_merged_px, n_spx, n_new_spx);

  count = 0;
  id_count = std::vector<int>(n_spx, 0);
  for (int j=graph.spx_s.size()-1; j>=0; --j){
    const auto &v = graph.spx_s[j];
    if (!v.is_valid)
      continue;
    assert(j == v.id);
    const auto &e = graph.adjacent[j];
    assert(!e.count(j));
    const int id = shrink_id_map[j];
    assert(id >= 0);
    for (auto px : v.pxs)
      label.at<int>(px.y, px.x) = id;
    count += v.pxs.size();
    id_count[id] += v.pxs.size();
  }
  assert(count == (label.rows * label.cols));
  for (int j=id_count.size()-1; j>=0; --j)
    assert(id_count[j] > num_max_merged_px || id_count[j] <= 0);
}


void
initialize(cv::Mat_<int> &label,
           std::vector<SuperPixel> &spx_s,
           const cv::Mat_<cv::Vec3b> &rgb_img,
           const cv::Mat_<uchar> &edge,
           const HyperParams &params) {
  const int diameter = std::sqrt(rgb_img.rows * rgb_img.cols / params.expect_spx_num);
  const int n = ((rgb_img.rows + diameter - 1) / diameter) * ((rgb_img.cols + diameter - 1) / diameter);
  spx_s.resize(n);
  int count=0;
  for (int y_start=0; y_start<rgb_img.rows; y_start+=diameter)
    for (int x_start=0; x_start<rgb_img.cols; x_start+=diameter){
      int y_end=std::min(y_start+diameter, rgb_img.rows),
          x_end=std::min(x_start+diameter, rgb_img.cols);
      cv::Vec<F64, 3> rgb{ 0, 0, 0 };
      cv::Vec<F64, 2> xy{ 0, 0 };
      int n_px = (y_end - y_start) * (x_end - x_start);
      for (int y=y_start; y<y_end; ++y)
        for(int x=x_start; x<x_end; ++x){
          auto px_rgb = rgb_img.at<cv::Vec3b>(y, x);
          for (int i=0; i<3; ++i)
            rgb[i] += px_rgb[i];
          xy[0] += x;
          xy[1] += y;
          label.at<int>(y, x) = count;
        }
      assert(n_px >= 1);
      spx_s[count].rgb_mean = {
          clamp2uchar(rgb[0] / (F64) n_px),
          clamp2uchar(rgb[1] / (F64) n_px),
          clamp2uchar(rgb[2] / (F64) n_px)
      };
      spx_s[count].xy_mean = {
          (int) (xy[0] / (F64) n_px),
          (int) (xy[1] / (F64) n_px)
      };
      spx_s[count].n_px = n_px;
      ++count;
    }
  assert(count == n);

  initialize_hyper_params(params, diameter);

  //在迭代过程中做分割，可以使方法快速收敛
  label = split_by_edge(label, edge, params.verbose);
  merge_small_spx(label, rgb_img, (diameter*diameter/9-1), params.verbose);
  update_spx_from_label(spx_s, label, rgb_img);
}


cv::Mat_<int>
ccs_(const cv::Mat_<cv::Vec3b> &rgb_img,
     const HyperParams &params){
  cv::Mat_<cv::Vec3b> img;
  if (params.rgb2lab) {
    img = rgb2lab(rgb_img, true);
    if (params.verbose)
      save_image("input_image_lab.png", img);
  } else {
    img = rgb_img;
  }
  const cv::Mat_<uchar>
      grad = rgb2grad(rgb_img),
      edge = edge_mat(rgb2gray(rgb_img));
  if (params.verbose) {
    save_image("input_image_grad.png", grad);
    save_image("input_image_edge.png", edge);
  }

  cv::Mat_<int> label( img.rows, img.cols, -1 );
  std::vector<SuperPixel> spx_s;
  initialize(label, spx_s, img, edge, params);

  if (params.verbose) {
    save_edge_map(
        str_printf(512, "%s_init_mat.png", __func__),
        label);
    save_segmentation_map(
        str_printf(512, "%s_init_mat_color.png", __func__),
        label);
    std::vector<cv::Point2i> spx_pos;
    spx_pos.reserve(spx_s.size());
    for (int j=spx_s.size()-1; j>=0; --j)
      if (spx_s[j].n_px > 0)
        spx_pos.emplace_back(spx_s[j].xy_mean);
    save_super_pixel(
        str_printf(512, "%s_init_super_pixel.png", __func__),
        spx_pos, label, rgb_img);
  }

  for (int iter=0; iter<params.max_iter_num; ++iter){
    if (params.verbose)
      printf("start at iterations-%d...\n", iter);

    auto new_label = label.clone();
    std::vector<cv::Point2i> boundary_px = find_boundary_px(label, edge, false);
    while (!boundary_px.empty()) {
      auto px = boundary_px.back();
      boundary_px.pop_back();

      //优化
      auto adj_spx = adjacent_spx(px, label);
      assert(!adj_spx.empty());
      int min_spx_id = -1;
      F32 min_spx_dist = std::numeric_limits<float>::max();
      for (int id : adj_spx){
        F32 d = dist_px2spx(px, spx_s[id], img, grad, params);
        if (d < min_spx_dist){
          min_spx_id   = id;
          min_spx_dist = d;
        }
      }
      new_label.at<int>(px.y, px.x) = min_spx_id;
    }

    update_spx_from_label(spx_s, new_label, img);
    label = new_label;

    if (params.verbose) {
      save_edge_map(
          str_printf(512, "%s_mat_%d.png", __func__, iter),
          label);
      save_segmentation_map(
          str_printf(512, "%s_mat_%d_color.png", __func__, iter),
          label);
      std::vector<cv::Point2i> spx_pos;
      spx_pos.reserve(spx_s.size());
      for (int j=spx_s.size()-1; j>=0; --j)
        if (spx_s[j].n_px > 0)
          spx_pos.emplace_back(spx_s[j].xy_mean);
      save_super_pixel(
          str_printf(512, "%s_img_%d.png", __func__, iter),
          spx_pos, label, rgb_img);
    }
  }

  //label中可能存在空间上不相邻，但是ID一致的多个簇

#if 0
  const int n_max_merged_px = params.expect_spx_diameter*params.expect_spx_diameter/9 - 1;
  merge_small_spx(label, img, n_max_merged_px, params.verbose);
  update_spx_from_label(spx_s, label, img);
  if (params.verbose) {
    save_edge_map(
        str_printf(512, "%s_mat_final.png", __func__),
        label);
    save_segmentation_map(
        str_printf(512, "%s_mat_final_color.png", __func__),
        label);
    std::vector<cv::Point2i> spx_pos;
    spx_pos.reserve(spx_s.size());
    for (int j=spx_s.size()-1; j>=0; --j)
      if (spx_s[j].n_px > 0)
        spx_pos.emplace_back(spx_s[j].xy_mean);
    save_super_pixel(
        str_printf(512, "%s_img_final.png", __func__),
        spx_pos, label, rgb_img);
  }
#endif

  return label;
}


cv::Mat_<int>
ccs(const cv::Mat_<cv::Vec3b> &rgb_img,
    const HyperParams &params) {
  cv::Mat_<int> result;
  try {
    namespace time = std::chrono;
    auto t0 = time::steady_clock::now();
    result = ccs_(rgb_img, params);
    auto t1 = time::steady_clock::now();
    float used_sec = time::duration_cast<time::milliseconds>(t1 - t0).count() * 1e-3f;
    if (params.verbose)
      printf("%s algorithm cost %.3f second\n", __func__, used_sec);
  } catch (const std::exception &e) {
    printf("\nfunction-%s: Catch Error %s\n", __func__, e.what());
    fflush(stdout);
    throw ;
  } catch (...) {
    printf("\nfunction-%s: Unknown Error!\n", __func__);
    fflush(stdout);
    throw ;
  }
  return result;
}

