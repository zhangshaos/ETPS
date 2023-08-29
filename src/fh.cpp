#include "fh.h"
#include "self_check_float.h"
#include "print_cv_mat.h"
#include "preprocess.h"
#include "throw_exception.h"
#include "disjoint_set.h"
#include <numeric>
#undef NDEBUG
#include <cassert>


struct Vertex{
  int ind{-1};//-1 means invalid
  cv::Point2i xy{0,0};
};

struct Edge{
  int ind{-1};
  int adj_ind{-1};//-1 means invalid
  F32 weight{std::numeric_limits<float>::max()};
};

//最小生成树
struct MST{
  F32 max_edge_weight = 0;
  F32 mean_edge_weight = 0;
  F32 var_edge_weight = 0;
  F32 std_edge_weight = 0;
  cv::Vec3b mean_rgb{0, 0, 0};
  int n_vertex = 0;
};

struct Graph{
  std::vector<Vertex> vertices;
  std::vector<std::array<Edge, 4> > adj_edges;
  DisjointSet vertex_id;
  std::vector<MST> MST_forest;
};

inline
F32
dist_rgb(cv::Vec3b a, cv::Vec3b b){
  F32 result = 0;
  for (int i=0; i<3; ++i){
    F32 d = (F32)a[i] - (F32)b[i];
    result += d*d;
  }
  return result;
}

inline
F32
dist_rgb_patch(const int x0, const int y0,
               const int x1, const int y1,
               const cv::Mat_<cv::Vec3b> &rgb_img,
               const int r){
  F32 d_sum = 0;
  int n = 0;
  for (int dy=-r; dy<=r; ++dy)
    for (int dx=-r; dx<=r; ++dx){
      const int _y0 = y0 + dy, _x0 = x0 + dx,
                _y1 = y1 + dy, _x1 = x1 + dx;
      if (_y0 < 0 || _y0 >= rgb_img.rows ||
          _x0 < 0 || _x0 >= rgb_img.cols ||
          _y1 < 0 || _y1 >= rgb_img.rows ||
          _x1 < 0 || _x1 >= rgb_img.cols)
        continue;
      auto v0 = rgb_img.at<cv::Vec3b>(_y0, _x0),
           v1 = rgb_img.at<cv::Vec3b>(_y1, _x1);
      F32 d = dist_rgb(v0, v1);
      d_sum += d;
      ++n;
    }
  assert(n >= 1);
  return (d_sum / (F32)n);
}

F32
weight_edge(int x0, int y0, int x1, int y1,
            const cv::Mat_<cv::Vec3b> &rgb_img,
            const HyperParams &params){
  assert(y0 >= 0 && y0 < rgb_img.rows);
  assert(x0 >= 0 && x0 < rgb_img.cols);
  assert(y1 >= 0 && y1 < rgb_img.rows);
  assert(x1 >= 0 && x1 < rgb_img.cols);
  F32 d = 0;
#define PATCH_DIST 0
#if PATCH_DIST
    d = std::sqrt(dist_rgb_patch(x0, y0, x1, y1, rgb_img, params.radius));
#else
    auto a = rgb_img.at<cv::Vec3b>(y0, x0),
         b = rgb_img.at<cv::Vec3b>(y1, x1);
    d = std::sqrt(dist_rgb(a, b));
#endif
  return d;
}

void
initialize(Graph &graph,
           const cv::Mat_<cv::Vec3b> &rgb_img,
           const HyperParams &params){
  constexpr int offset[4][2] = {
      { 0, -1},
      { 1,  0},
      { 0,  1},
      { -1, 0},
  };
  const int n = rgb_img.rows * rgb_img.cols;
#define xy2ind(x, y) ((y)*rgb_img.cols + (x))

  graph.vertices.resize(n);
  graph.adj_edges.resize(n);//无向边的数量 = 实际边的数量 * 2
  for (int y=0; y<rgb_img.rows; ++y)
    for (int x=0; x<rgb_img.cols; ++x){
      int i = xy2ind(x, y);
      auto &v = graph.vertices[i];
      v.ind = i;
      v.xy = cv::Point2i{x, y};
      auto &e = graph.adj_edges[i];
      for (int j=std::size(offset); j>=0; --j){
        int x1 = x + offset[j][0],
            y1 = y + offset[j][1];
        if (x1 < 0 || x1 >= rgb_img.cols ||
            y1 < 0 || y1 >= rgb_img.rows)
          continue;
        int i1 = xy2ind(x1, y1);
        e[j].ind     = i;
        e[j].adj_ind = i1;
        e[j].weight = weight_edge(x, y, x1, y1, rgb_img, params);
      }
    }

#define NON_MAX_SUPPRESSION 1
#if NON_MAX_SUPPRESSION
#define max_w(wa, wb) ((wa) > (wb) ? (wa) : (wb))
  //非极大值抑制=>极大值提升，解决边缘模糊梯度问题（即边缘不太明显）
  for (int y=0; y<rgb_img.rows; ++y)
    for (int x=0; x<rgb_img.cols; ++x){
      int i = xy2ind(x, y);
      auto &e0 = graph.adj_edges[i][0],
           &e2 = graph.adj_edges[i][2];
      if (e0.adj_ind >= 0 && e2.adj_ind >= 0)
        max_w(e0.weight, e2.weight) *= (F32)1.5;
      auto &e1 = graph.adj_edges[i][1],
           &e3 = graph.adj_edges[i][3];
      if (e1.adj_ind >= 0 && e3.adj_ind >= 0)
        max_w(e1.weight, e3.weight) *= (F32)1.5;
      //注意，由于无向图中两条边对应一个实际边
    }
#endif

  for (int y=0; y<rgb_img.rows; ++y)
    for (int x=0; x<rgb_img.cols; ++x) {
      int i = xy2ind(x, y);
      graph.vertex_id.try_add_class(i);
    }

  graph.MST_forest.resize(n);//刚开始，每个顶点作为一棵MST
  for (int y=0; y<rgb_img.rows; ++y)
    for (int x=0; x<rgb_img.cols; ++x) {
      int i = xy2ind(x, y);
      assert(i == graph.vertex_id.find_root_class(i));
      auto &mst = graph.MST_forest[i];
      mst.n_vertex        = 1;
      mst.max_edge_weight = 0;
      mst.mean_rgb        = rgb_img.at<cv::Vec3b>(y, x);

      const auto &adj_e = graph.adj_edges[i];
      F32 mean_w = 0;
      F32 max_w = -1;
      int max_w_ind = -1;
      int n = 1;//包括自己和自己（距离为0）
      for (int j=std::size(offset); j>=0; --j){
        if (adj_e[j].adj_ind < 0)
          continue;
        F32 w = adj_e[j].weight;
        mean_w += w;
        if (w > max_w) {
          max_w = w;
          max_w_ind = j;
        }
        ++n;
      }
      assert(n > 1);
      mean_w -= max_w;//去掉一个最大值
      --n;
      mean_w /= (F32)n;
      F32 var_w = 0;
      for (int j=std::size(offset); j>=0; --j){
        if (adj_e[j].adj_ind < 0 || j == max_w_ind)
          continue;
        F32 w = adj_e[j].weight;
        F32 d = w - mean_w;
        var_w += d*d;
      }
      var_w /= (F32)n;
      mst.mean_edge_weight = mean_w;
      mst.var_edge_weight  = var_w;
      mst.std_edge_weight  = std::sqrt(var_w);
    }
}

F32
weight_threshold_of_two_components(int id0, int id1,
                                   const Graph &graph,
                                   const HyperParams &params){
  assert(graph.vertex_id.is_root_class(id0));
  assert(graph.vertex_id.is_root_class(id1));
  const auto &mst0 = graph.MST_forest[id0],
             &mst1 = graph.MST_forest[id1];
  assert(mst0.n_vertex > 0 && mst1.n_vertex > 0);
#if 0
  F32 v0 = mst0.max_edge_weight + (F32) params.k / (F32) mst0.n_vertex,
      v1 = mst1.max_edge_weight + (F32) params.k / (F32) mst1.n_vertex;
#endif
  F32 d = std::min(mst0.max_edge_weight, mst1.max_edge_weight);
  d += dist_rgb(mst0.mean_rgb, mst1.mean_rgb);
  return d;
}

cv::Vec3b
mean_rgb_MST(const MST &l, const MST &r){
  F32 a0 = (F32)l.n_vertex / (F32)(l.n_vertex + r.n_vertex),
      a1 = (F32)r.n_vertex / (F32)(l.n_vertex + r.n_vertex);
  F32 _r = (F32)l.mean_rgb[0] * a0 + (F32)r.mean_rgb[0] * a1,
      _g = (F32)l.mean_rgb[1] * a0 + (F32)r.mean_rgb[1] * a1,
      _b = (F32)l.mean_rgb[2] * a0 + (F32)r.mean_rgb[2] * a1;
  return { (uchar)_r, (uchar)_g, (uchar)_b };
}

bool
try_merge(Graph &graph,
          int composed_edge,
          const HyperParams &params){
  const Edge edge = graph.adj_edges[composed_edge / 4][composed_edge % 4];
  assert(edge.ind >= 0 && edge.adj_ind >= 0);
  const int id0 = graph.vertex_id.find_root_class(edge.ind),
            id1 = graph.vertex_id.find_root_class(edge.adj_ind);
  if (id0 == id1)
    return false;

  F32 min_edge_weight = edge.weight;//必须要先对边权重进行排序
  F32 max_compo_weight = weight_threshold_of_two_components(id0, id1, graph, params);
  if (min_edge_weight > max_compo_weight)
    return false;

  graph.vertex_id.union_class(id0, id1);
  auto &mst0 = graph.MST_forest[id0];
  auto &mst1 = graph.MST_forest[id1];
  //合并之后的MST就是原来的两个MST加上它们之间的最小边min_edge_weight
  const F32 MST_max_weight     = min_edge_weight;//边权重升序排列
  const int MST_n_vertex       = mst0.n_vertex + mst1.n_vertex;
  const cv::Vec3b MST_mean_rgb = mean_rgb_MST(mst0, mst1);
  const F32 a_sum = std::max(1, mst0.n_vertex-1) + std::max(1, mst1.n_vertex-1) + 1;
  const F32 a0 = (F32)std::max(1, mst0.n_vertex-1) / a_sum,
            a1 = (F32)std::max(1, mst1.n_vertex-1) / a_sum,
            a2 = (F32)1                            / a_sum;
  const F32 MST_mean_w  = a0 * mst0.mean_edge_weight +
                          a1 * mst1.mean_edge_weight +
                          a2 * min_edge_weight;
  const F32 d0 = MST_mean_w - mst0.mean_edge_weight,
            d1 = MST_mean_w - mst1.mean_edge_weight,
            d2 = MST_mean_w - min_edge_weight;
  const F32 MST_var_w   = a0 * (mst0.var_edge_weight + d0*d0) +
                          a1 * (mst1.var_edge_weight + d1*d1) +
                          a2 * ((F32)0             + d2*d2);
  mst0.n_vertex         = mst1.n_vertex         = MST_n_vertex;
  mst0.mean_rgb         = mst1.mean_rgb         = MST_mean_rgb;
  mst0.max_edge_weight  = mst1.max_edge_weight  = MST_max_weight;
  mst0.mean_edge_weight = mst1.mean_edge_weight = MST_mean_w;
  mst0.var_edge_weight  = mst1.var_edge_weight  = MST_var_w;
  mst0.std_edge_weight  = mst1.std_edge_weight  = std::sqrt(MST_var_w);
  // 由于边缘的权重太大，会对早期合并产生问题，因此需要先合并权重小的。
  return true;
}

cv::Mat_<int>
graph2label(const Graph &graph, const int h, const int w){
  assert(graph.vertices.size() == h*w);
  cv::Mat_<int> result(h, w, -1);
  for (int y=0; y<h; ++y)
    for (int x=0; x<w; ++x){
      int i = y*w + x;
      assert(graph.vertex_id.contain_class(i));
      int id = graph.vertex_id.find_root_class(i);
      result.at<int>(y, x) = id;
    }
  return result;
}

cv::Mat_<int>
fh_(const cv::Mat_<cv::Vec3b> &rgb_img,
    const HyperParams &params){
  cv::Mat_<cv::Vec3b> img;
  if (params.rgb2lab) {
    img = rgb2lab(rgb_img);
    if (params.verbose)
      save_image("input_image_lab.png", img);
  } else {
    img = rgb_img;
  }

  Graph graph;
  initialize(graph, img, params);
  std::vector<int> sorted_edge_ind(graph.vertices.size() * 4, -1);
#define ind0(e) ((e) / 4)
#define ind1(e) ((e) % 4)
  std::iota(sorted_edge_ind.begin(), sorted_edge_ind.end(), 0);
  std::sort(sorted_edge_ind.begin(), sorted_edge_ind.end(),
            [&edges=graph.adj_edges](int e0, int e1){
    return edges[ind0(e0)][ind1(e0)].weight < edges[ind0(e1)][ind1(e1)].weight;
  });

  do {
    params._max_num_vertex = img.rows * img.cols;
    int max_e = *std::find_if(sorted_edge_ind.rbegin(),
                              sorted_edge_ind.rend(),
                              [&adj_edges=graph.adj_edges](int e){
                                return adj_edges[ind0(e)][ind1(e)].ind >= 0;
                              });
    const Edge max_edge = graph.adj_edges[ind0(max_e)][ind1(max_e)];
    assert(max_edge.ind >= 0 && max_edge.adj_ind >= 0);
    params._max_edge_weight = max_edge.weight;
  } while (0);

  int step = 1;
  const int max_step = (4 * img.rows * img.cols - 2 * (img.rows + img.cols)) / 2;
  const int n_v_step = params.visualize_step_ratio * max_step;
  for(int e : sorted_edge_ind){
    const Edge edge = graph.adj_edges[ind0(e)][ind1(e)];
    if (edge.adj_ind < 0)
      continue;
    assert(edge.ind >= 0);

    try_merge(graph, e, params);

    if (params.verbose && (step % n_v_step == 0)) {
      auto label = graph2label(graph, img.rows, img.cols);
      save_image_with_segmentation(str_printf(512, "seg_%d.png", step), rgb_img, label);
      save_edge_map(str_printf(512, "seg_edge_%d.png", step), label);
      save_segmentation_map(str_printf(512, "seg_color_%d.png", step), label);
    }
    ++step;
  }

  auto label = graph2label(graph, img.rows, img.cols);
  if (params.verbose){
    save_image_with_segmentation("seg_final.png", rgb_img, label);
    save_edge_map("seg_edge_final.png", label);
    save_segmentation_map("seg_color_final.png", label);
  }
  return label;
}

cv::Mat_<int>
fh(const cv::Mat_<cv::Vec3b> &rgb_img,
   const HyperParams &params) {
  cv::Mat_<int> result;
  try {
    result = fh_(rgb_img, params);
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

