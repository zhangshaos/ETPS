#include "connectivity.h"
#include "disjoint_set.h"
#include "self_check_float.h"
#include <set>


inline
bool
continued_in_edge_mat(const cv::Mat_<uchar> &edge,
                      int x0, int y0, int x1, int y1){
  return !edge.at<uchar>(y0, x0) || edge.at<uchar>(y1, x1);
}


cv::Mat_<int>
check_connectivity(const cv::Mat_<int> &label,
                   const cv::Mat_<uchar> &edge,
                   const HyperParams &param){
  constexpr int offset[][2] = {
      {-1, -1},//dx,dy
      { 0, -1},
      { 1, -1},
      {-1,  0},
  };

  DisjointSet dj_set;
  int start_id = 0;
  cv::Mat_<int> result(label.rows, label.cols, -1);

  //first pass
  for (int y0=0; y0<label.rows; ++y0)
    for (int x0=0; x0<label.cols; ++x0){
      const int spx_id0  = label.at<int>(y0, x0);
      //测试当前像素和过去像素的连通性
      std::set<int> connected;
      for (int j=std::size(offset)-1; j>=0; --j){
        int y = y0 + offset[j][1],
            x = x0 + offset[j][0];
        if (y < 0 || y >= label.rows ||
            x < 0 || x >= label.cols)
          continue;
        const int spx_id1 = label.at<int>(y, x);
        if (spx_id1 != spx_id0)
          continue;
        if (continued_in_edge_mat(edge, x0, y0, x, y)) {
          int id = result.at<int>(y, x);
          assert(id >= 0);
          id = dj_set.find_root_class(id);
          assert(id >= 0);
          connected.emplace(id);
        }
      }

      int cur_id = -1;
      if(connected.empty()){
        cur_id = start_id++;
        dj_set.try_add_class(cur_id);
        result.at<int>(y0, x0) = cur_id;
      } else {
        cur_id = *connected.begin();//最小值
        if(connected.size() > 1){
          auto it = connected.begin();
          while (++it != connected.end())
            dj_set.union_class(cur_id, *it);//合并结果是最小的那一个
        }
        result.at<int>(y0, x0) = cur_id;
      }
      assert(cur_id >= 0);
    }

  //second pass
  std::vector<uint32_t> shrink_parent;
  dj_set.shrink_parent(&shrink_parent);
  for (int y=0; y<label.rows; ++y)
    for (int x=0; x<label.cols; ++x){
      int c = result.at<int>(y, x);
      assert(c >= 0);
      c = (int) shrink_parent[c];
      result.at<int>(y, x) = c;
    }

  return result;
}


#if 0
inline
bool
continued_grad(F64 grad0, F64 grad1, const HyperParams &params){
  return std::abs( float(grad0 - grad1) ) <= params.continued_grad_scale * 255.f;
}


cv::Mat_<int>
check_connectivity(const cv::Mat_<int> &label,
                   const cv::Mat_<uchar> &grad,
                   const HyperParams &param){
  constexpr int offset[][2] = {
      {-1, -1},//dx,dy
      { 0, -1},
      { 1, -1},
      {-1,  0},
  };

  DisjointSet dj_set;
  struct Grad_Count{
    F64 grad = 0;
    int count = 0;
  };
  std::unordered_map<int, Grad_Count> grad_counter;
  grad_counter.reserve(param.expect_spx_num * 2);
  int start_id = 0;
  cv::Mat_<int> result(label.rows, label.cols, -1);

  //first pass
  for (int y0=0; y0<label.rows; ++y0)
    for (int x0=0; x0<label.cols; ++x0){
      const int spx_id0  = label.at<int>(y0, x0);
      const uchar grad0  = grad.at<uchar>(y0, x0);
      //测试当前像素和过去像素的连通性
      std::set<int> connected;
      for (int j=std::size(offset)-1; j>=0; --j){
        int y = y0 + offset[j][1],
            x = x0 + offset[j][0];
        if (y < 0 || y >= label.rows ||
            x < 0 || x >= label.cols)
          continue;
        const int spx_id1 = label.at<int>(y, x);
        if (spx_id1 != spx_id0)
          continue;
        int id = result.at<int>(y, x);
        assert(id >= 0);
        id = dj_set.find_root_class(id);
        assert(id >= 0);
        F64 grad_mean = grad_counter.at(id).grad / (F64) grad_counter.at(id).count;
        if (continued_grad((F64) grad0, grad_mean, param))
          connected.emplace(id);
      }

      int cur_id = -1;
      if(connected.empty()){
        cur_id = start_id++;
        dj_set.try_add_class(cur_id);
        result.at<int>(y0, x0) = cur_id;
      } else {
        cur_id = *connected.begin();//最小值
        if(connected.size() > 1){
          //尝试合并可以合并的
          for (int id : connected){
            if (id == cur_id)
              continue;
            F64 m0 = grad_counter.at(cur_id).grad / (F64) grad_counter.at(cur_id).count,
                m1 = grad_counter.at(id).grad / (F64) grad_counter.at(id).count;
            if (!continued_grad(m0, m1, param))
              continue;
            dj_set.union_class(cur_id, id);//合并结果是最小的那一个
            int count_ = grad_counter.at(cur_id).count + grad_counter.at(id).count;
            F64 grad_  = grad_counter.at(cur_id).grad + grad_counter.at(id).grad;
            grad_counter.at(cur_id).count = grad_counter.at(id).count = count_;
            grad_counter.at(cur_id).grad  = grad_counter.at(id).grad  = grad_;
          }
        }
        result.at<int>(y0, x0) = cur_id;
      }

      assert(cur_id >= 0);
      if (grad_counter.count(cur_id)){
        grad_counter[cur_id].count++;
        grad_counter[cur_id].grad += (F64) grad0;
      } else {
        grad_counter[cur_id].count = 1;
        grad_counter[cur_id].grad = grad0;
      }
    }

  //second pass
  std::vector<uint32_t> shrink_parent;
  dj_set.shrink_parent(&shrink_parent);
  for (int y=0; y<label.rows; ++y)
    for (int x=0; x<label.cols; ++x){
      int c = result.at<int>(y, x);
      assert(c >= 0);
      c = (int) shrink_parent[c];
      result.at<int>(y, x) = c;
    }

  return result;
}
#endif


cv::Mat_<uchar>
edge_mat(const cv::Mat_<uchar> &grad,
         const HyperParams &params){
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

  F64 mean_grad = 0;
  int count = 0;
  for (int y=0; y<grad.rows; ++y)
    for (int x=0; x<grad.cols; ++x) {
      uchar v = grad.at<uchar>(y, x);
      if (v <= 0)
        continue;
      mean_grad += (F64)v;
      ++count;
    }
  mean_grad = count > 0 ? mean_grad / (F64)count : (F64)0;
  cv::Mat_<uchar> result(grad.rows, grad.cols, (uchar)0);
  for (int y=0; y<grad.rows; ++y)
    for (int x=0; x<grad.cols; ++x) {
      uchar v = grad.at<uchar>(y, x);
      if (v <= mean_grad)
        continue;
      result.at<uchar>(y, x) = 0xff;
    }

  //删除很短的边
  cv::Mat_<uchar> extended(result.rows, result.cols, (uchar)0);
  for (int y0=0; y0<result.rows; ++y0)
    for (int x0=0; x0<result.cols; ++x0){
      if (result.at<uchar>(y0, x0)<=0 || extended.at<uchar>(y0, x0)>0)
        continue;
      //DFS统计边大小
      std::vector<cv::Point2i> accessed;
      std::vector<cv::Point2i> extend_stack;
      extend_stack.emplace_back(x0, y0);
      extended.at<uchar>(y0, x0) = 1;
      while (!extend_stack.empty()){
        auto top = extend_stack.back();
        extend_stack.pop_back();
        accessed.emplace_back(top);
        assert(extended.at<uchar>(top.y, top.x));

        for (int j=std::size(offset)-1; j>=0; --j){
          int y = top.y + offset[j][1],
              x = top.x + offset[j][0];
          if(y < 0 || y >= result.rows ||
             x < 0 || x >= result.cols ||
             result.at<uchar>(y, x)<=0 || extended.at<uchar>(y, x)>0)
            continue;
          extend_stack.emplace_back(x, y);
          extended.at<uchar>(y, x) = 1;
        }
      }
      if (accessed.size() < params.min_edge_threshold)
        for (auto p : accessed)
          result.at<uchar>(p.y, p.x) = 0;
    }
  return result;
}