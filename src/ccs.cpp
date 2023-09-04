#include "ccs.h"
#include "self_check_float.h"
#include "print_cv_mat.h"
#include "preprocess.h"
#include "throw_exception.h"
#include "connectivity.h"
#include "geodesic_distance.h"
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
  F32 max_d_rgb             = 25.f * 25.f;
  F32 max_d_spatial         = 2.f * diameter * 0.5f * diameter * 0.5f;
  params.spatial_scale      *= ((float)max_d_rgb / (float)max_d_spatial);
  params.min_edge_threshold =  (0.5f * diameter);
  params.max_iter_num       =  2 * diameter;
  printf("%s(): spatial_scale=%f, min_edge_threshold=%d, max_iter_num=%d\n",
         __func__, params.spatial_scale, params.min_edge_threshold, params.max_iter_num);
}

void
initialize(cv::Mat_<int> &label,
           std::vector<SuperPixel> &spx_s,
           const cv::Mat_<cv::Vec3b> &rgb_img,
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


std::set<int>
adjacent_spx(cv::Point2i px,
             const cv::Mat_<int> &label) {
  std::set<int> adj_spx;
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
  for (int t=std::size(offset); t>=0; --t){
    int y = px.y + offset[t][1],
        x = px.x + offset[t][0];
    if (y < 0 || y >= label.rows || x < 0 || x >= label.cols)
      continue;
    int id = label.at<int>(y, x);
    adj_spx.emplace(id);
  }
  return adj_spx;
}

//#pragma GCC optimize("O0")
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
    int r = rand() % (params.max_iter_num * params.expect_spx_num);
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
//#pragma GCC optimize("O2")

void
update_spx(std::vector<SuperPixel> &spx_s,
           const cv::Mat_<int> &label,
           const cv::Mat_<cv::Vec3b> &rgb_img,
           const HyperParams &params){
  struct RGB_XY_Count{
    cv::Vec<F64, 3> rgb{ 0, 0, 0 };
    cv::Vec<F64, 2> xy{ 0, 0, 0 };
    int count{ 0 };
  };
  std::vector<RGB_XY_Count> px_counter;
  px_counter.resize(spx_s.size());

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

  cv::Mat_<int> label( img.rows, img.cols, -1 );
  std::vector<SuperPixel> spx_s;
  initialize(label, spx_s, img, params);
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

  cv::Mat_<uchar> grad = rgb2grad(rgb_img),
                  edge = edge_mat(grad, params);
  if (params.verbose) {
    save_image("input_image_grad.png", grad);
    save_image("input_image_edge.png", edge);
  }

  for (int iter=0; iter<params.max_iter_num; ++iter){
    if (params.verbose){
      printf("start at iterations-%d...\n", iter);
      fflush(stdout);
    }

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

    update_spx(spx_s, new_label, img, params);
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

  //fixme not work
  label = check_connectivity(label, edge, params);
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

  return label;
}


cv::Mat_<int>
ccs(const cv::Mat_<cv::Vec3b> &rgb_img,
    const HyperParams &params) {
  cv::Mat_<int> result;
  try {
    result = ccs_(rgb_img, params);
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

