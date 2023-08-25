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
  int n_vertex = 0;
};

struct Graph{
  std::vector<Vertex> vertices;
  std::vector<std::array<Edge, 4> > adj_edges;
  DisjointSet vertex_id;
  std::vector<MST> MST_forest;
};

F32
dist_rgb(cv::Vec3b a, cv::Vec3b b){
  F32 result = 0;
  for (int i=0; i<3; ++i){
    F32 d = (F32)a[i] - (F32)b[i];
    result += d*d;
  }
  return result;
}

F32
weight_edge(int x0, int y0, int x1, int y1,
            const cv::Mat_<cv::Vec3b> &rgb_img){
  assert(y0 >= 0 && y0 < rgb_img.rows);
  assert(x0 >= 0 && x0 < rgb_img.cols);
  assert(y1 >= 0 && y1 < rgb_img.rows);
  assert(x1 >= 0 && x1 < rgb_img.cols);
  auto a = rgb_img.at<cv::Vec3b>(y0, x0),
       b = rgb_img.at<cv::Vec3b>(y1, x1);
  F32 d = dist_rgb(a, b);
  //Future use patch will be better?
  return d;
}

void
initialize(Graph &graph,
           const cv::Mat_<cv::Vec3b> &rgb_img){
  constexpr int offset[4][2] = {
      { 0, -1},
      { 1,  0},
      { 0,  1},
      { -1, 0},
  };
  const int n = rgb_img.rows * rgb_img.cols;
  graph.vertices.resize(n);
  graph.adj_edges.resize(n);
  graph.MST_forest.resize(n);//刚开始，每个顶点作为一棵MST
  for (int y=0; y<rgb_img.rows; ++y)
    for (int x=0; x<rgb_img.cols; ++x){
#define xy2ind(x, y) ((y)*rgb_img.cols + (x))
      int i = xy2ind(x, y);
      auto &v = graph.vertices[i];
      v.ind = i;
      v.xy = cv::Point2i{x, y};
      graph.vertex_id.try_add_class(i);
      assert(i == graph.vertex_id.find_root_class(i));
      graph.MST_forest[i].n_vertex = 1;
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
        e[j].weight = weight_edge(x, y, x1, y1, rgb_img);
      }
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
//  F32 v0 = (mst0.max_edge_weight / (F32)params._max_edge_weight) +
//           ((F32)mst0.n_vertex / (F32)params._max_num_vertex) * (F32)params.k;
//  v0 *= (F32)params._max_edge_weight;
  F32 v0 = mst0.max_edge_weight + (F32) params.k / (F32) mst0.n_vertex,
      v1 = mst1.max_edge_weight + (F32) params.k / (F32) mst1.n_vertex;
  return std::min(v0, v1);
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
  F32 MST_max_weight = min_edge_weight;//边权重升序排列
  int MST_n_vertex   = mst0.n_vertex + mst1.n_vertex;
  mst0.n_vertex        = mst1.n_vertex        = MST_n_vertex;
  mst0.max_edge_weight = mst1.max_edge_weight = MST_max_weight;
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
  initialize(graph, img);
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
    int max_e = sorted_edge_ind.back();
    const Edge max_edge = graph.adj_edges[ind0(max_e)][ind1(max_e)];
    params._max_edge_weight = max_edge.weight;
  } while (0);

  for(int e : sorted_edge_ind){
    const Edge edge = graph.adj_edges[ind0(e)][ind1(e)];
    if (edge.adj_ind < 0)
      continue;
    assert(edge.ind >= 0);

    try_merge(graph, e, params);
  }

  auto label = graph2label(graph, img.rows, img.cols);
  if (params.verbose){
    save_image_with_segmentation("seg.png", rgb_img, label);
    save_edge_map("seg_edge.png", label);
    save_segmentation_map("seg_color.png", label);
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

