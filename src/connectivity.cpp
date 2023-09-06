#include "connectivity.h"
#include "disjoint_set.h"
#include "draw_line.h"
#include "print_cv_mat.h"
#include <opencv2/imgproc.hpp>


cv::Mat_<uchar>
edge_mat(const cv::Mat_<uchar> &img){
  auto lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);
  std::vector<cv::Vec4i> lines;
  lsd->detect(img, lines);
  cv::Mat_<uchar> result(img.rows, img.cols, (uchar)0);
  for (const auto &l : lines){
    cv::Point2i p0(l[0], l[1]), p1(l[2], l[3]);
    auto pxs = line_path(p0, p1);
    for (auto px : pxs)
      result.at<uchar>(px.y, px.x) = 0xff;
  }
  return result;
}


cv::Mat_<int>
split_by_edge(const cv::Mat_<int> &label,
              const cv::Mat_<uchar> &edge_,
              bool verbose){
  constexpr int offset[][2] = {
      {-1, -1},//dx,dy
      { 0, -1},
      { 1, -1},
      {-1,  0},
      { 1,  0},
      {-1,  1},
      { 0,  1},
      { 1,  1},
  };
  //避免8相邻时如下的穿透情况：
  //xo 和 ox
  //ox    xo
  auto edge = edge_.clone();
  for (int y0=0; y0<edge.rows-1; ++y0)
    for (int x0=0; x0<edge.cols-1; ++x0){
      int y1 = y0    , x1 = x0 + 1,
          y2 = y0 + 1, x2 = x0    ,
          y3 = y0 + 1, x3 = x0 + 1;
      bool on_edge0 = edge.at<uchar>(y0, x0),
           on_edge1 = edge.at<uchar>(y1, x1),
           on_edge2 = edge.at<uchar>(y2, x2),
           on_edge3 = edge.at<uchar>(y3, x3);
      if (on_edge0 && on_edge3 && !on_edge1 && !on_edge2)
        edge.at<uchar>(y2, x2) = 0xff;
      else if (!on_edge0 && !on_edge3 && on_edge1 && on_edge2)
        edge.at<uchar>(y3, x3) = 0xff;
    }
  //bugs：如果两条线段距离很近，通过上面的边缘修正方法会导致两条线段中间形成一个个小腔室，
  // 造成结果中出现一个个孤立的小点。
#if 0
  save_edge_map("foo.edge.png", edge);
#endif

  cv::Mat_<int> result(label.rows, label.cols, -1);
  int start_id = 0;

  cv::Mat_<uchar> extended(label.rows, label.cols, (uchar)0);
  for (int y0=0; y0<label.rows; ++y0)
    for (int x0=0; x0<label.cols; ++x0){
      if (extended.at<uchar>(y0, x0) || edge.at<uchar>(y0, x0))
        //如果边缘像素作为DFS起点，则只会找到这条边
        continue;
      const int id0 = label.at<int>(y0, x0);
      //DFS找到所有未被线段阻隔的像素
      std::vector<cv::Point2i> accessed;
      std::vector<cv::Point2i> extend_stack;
      extend_stack.emplace_back(x0, y0);
      extended.at<uchar>(y0, x0) = 1;
      while (!extend_stack.empty()){
        auto top = extend_stack.back();
        extend_stack.pop_back();
        accessed.emplace_back(top);
        assert(extended.at<uchar>(top.y, top.x));
        const bool on_edge = edge.at<uchar>(top.y, top.x);

        for (int j=std::size(offset)-1; j>=0; --j){
          int y = top.y + offset[j][1],
              x = top.x + offset[j][0];
          if(y < 0 || y >= label.rows ||
             x < 0 || x >= label.cols ||
             label.at<int>(y, x) != id0 ||
             extended.at<uchar>(y, x))
            continue;
          if(on_edge && !edge.at<uchar>(y, x))
            //处于边界上的像素，只能向其他边界像素扩展
            continue;
          extend_stack.emplace_back(x, y);
          extended.at<uchar>(y, x) = 1;
        }
      }
      for (auto p : accessed)
        result.at<int>(p.y, p.x) = start_id;
      ++start_id;
    }

  //处理全在边缘的一组相同label的像素（即，该label对应的所有像素都是边缘像素）
  for (int y0=0; y0<label.rows; ++y0)
    for (int x0=0; x0<label.cols; ++x0) {
      if (extended.at<uchar>(y0, x0))
        continue;
      assert(edge.at<uchar>(y0, x0));
      const int id0 = label.at<int>(y0, x0);
      //DFS找到所有像素
      std::vector<cv::Point2i> extend_stack;
      extend_stack.emplace_back(x0, y0);
      extended.at<uchar>(y0, x0) = 1;
      while (!extend_stack.empty()){
        auto top = extend_stack.back();
        extend_stack.pop_back();
        assert(extended.at<uchar>(top.y, top.x));
        assert(edge.at<uchar>(top.y, top.x));
        result.at<int>(top.y, top.x) = start_id;

        for (int j=std::size(offset)-1; j>=0; --j){
          int y = top.y + offset[j][1],
              x = top.x + offset[j][0];
          if(y < 0 || y >= label.rows ||
             x < 0 || x >= label.cols ||
             label.at<int>(y, x) != id0 ||
             extended.at<uchar>(y, x))
            continue;
          assert(edge.at<uchar>(y, x));
          extend_stack.emplace_back(x, y);
          extended.at<uchar>(y, x) = 1;
        }
      }
      ++start_id;
    }

#ifndef NDEBUG
  double min_v=0, max_v=0;
  cv::minMaxIdx(result, &min_v, &max_v);
  assert(min_v >= 0);
#endif

  if(verbose)
    printf("%s(): result %d clusters.\n", __func__, start_id);

  return result;
}


#if 0
cv::Mat_<int>
split_by_line(const cv::Mat_<int> &label,
              const std::vector<cv::Vec4i> &lines_segment){
  constexpr int offset[][2] = {
      {-1, -1},//dx,dy
      { 0, -1},
      { 1, -1},
      {-1,  0},
      { 1,  0},
      {-1,  1},
      { 0,  1},
      { 1,  1},
  };

  //加速结构：线段和超像素相交判断
  double min_v=0, max_v=0;
  cv::minMaxIdx(label, &min_v, &max_v);
  assert(min_v >= 0);
  const int n_spx = (int) max_v;
  struct BBox{
    float x_min=std::numeric_limits<float>::max(), x_max=0,
          y_min=std::numeric_limits<float>::max(), y_max=0;
    std::vector<cv::Vec3f> pxs;
  };
  std::vector<BBox> spx_bbox(n_spx);

  cv::Mat_<uchar> extended(label.rows, label.cols, (uchar)0);
  for (int y0=0; y0<label.rows; ++y0)
    for (int x0=0; x0<label.cols; ++x0){
      if (extended.at<uchar>(y0, x0))
        continue;
      const int id0 = label.at<int>(y0, x0);
      auto &bbox    = spx_bbox[id0];
      //DFS找到所有id相同的像素
      std::vector<cv::Point2i> extend_stack;
      extend_stack.emplace_back(x0, y0);
      extended.at<uchar>(y0, x0) = 1;
      while (!extend_stack.empty()){
        auto top = extend_stack.back();
        extend_stack.pop_back();
        assert(extended.at<uchar>(top.y, top.x));
        float x_ = top.x + 0.5f, y_ = top.y + 0.5f;
        bbox.pxs.emplace_back(x_, y_, 0.f);
        bbox.x_max = std::max(bbox.x_max, x_);
        bbox.x_min = std::min(bbox.x_min, x_);
        bbox.y_max = std::max(bbox.y_max, y_);
        bbox.y_min = std::min(bbox.y_min, y_);

        for (int j=std::size(offset)-1; j>=0; --j){
          int y = top.y + offset[j][1],
              x = top.x + offset[j][0];
          if(y < 0 || y >= label.rows ||
             x < 0 || x >= label.cols ||
             label.at<int>(y, x) != id0 ||
             extended.at<uchar>(y, x))
            continue;
          extend_stack.emplace_back(x, y);
          extended.at<uchar>(y, x) = 1;
        }
      }
    }

  //线段将相交的超像素包含的所有像素一分为二
  cv::Mat_<int> result(label.rows, label.cols, -1);
  int start_id = 0;

  //bugs: 无法将被四边形包围住的部分从整个超像素中分出来
  return result;
}
#endif