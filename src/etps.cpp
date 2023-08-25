#include "etps.h"
#include "self_check_float.h"
#include "print_cv_mat.h"
#include "preprocess.h"
#include "throw_exception.h"
#include <set>
#undef NDEBUG
#include <cassert>


inline
uchar
clamp2uchar(F64 v){
  return v > 0 ? (v < 255 ? (uchar) v : 255) : 0;
}

struct Cell{
  cv::Vec3b rgb{0, 0, 0};
  int spx_id = -1;
  int x_start = 0, x_end = -1,
      y_start = 0, y_end = -1;
  bool valid = false;

  Cell() = default;
  Cell(int spx_id_,
       int x_start_, int x_end_,
       int y_start_, int y_end_,
       bool valid_,
       const cv::Mat_<cv::Vec3b> &rgb_img)
   : spx_id(spx_id_),
     x_start(x_start_), x_end(x_end_),
     y_start(y_start_), y_end(y_end_),
     valid(valid_) {
    rgb = mean_rgb(rgb_img);
  }

  int
  area()const{
    return (y_end - y_start + 1) * (x_end - x_start + 1);
  }

  cv::Vec3b
  mean_rgb(const cv::Mat_<cv::Vec3b> &rgb_img)const{
    assert(valid);
    assert(y_end >= y_start && x_end >= x_start);
    cv::Vec<F64, 3> rgb_sum;
    for (int y=y_start; y<=y_end; ++y)
      for (int x=x_start; x<=x_end; ++x){
        auto rgb_ = rgb_img.at<cv::Vec3b>(y, x);
        for (int i=0; i<3; ++i)
          rgb_sum[i] += (F64) rgb_[i];
      }
    int n = (y_end - y_start + 1) * (x_end - x_start + 1);
    cv::Vec3b result;
    for (int i=0; i<3; ++i) {
      rgb_sum[i] /= (F64) n;
      result[i] = clamp2uchar(rgb_sum[i]);
    }
    return result;
  }
};

struct Grids{
  int level = -1;
  int cell_size = -1;//cell_size == 2^level
  std::vector<Cell> cell_s;
  int height = -1, width = -1;

  int
  idx_of_cell(int y, int x)const{
    return y * width + x;
  }
  cv::Point2i
  xy_of_cell(int i)const{
    int y = i / width,
        x = i % width;
    return {x, y};
  }

  std::set<int>
  adjacent_cells(int cell)const{
    std::set<int> adj_cells;
    //根据split_grids()，出现只有两个or一个子cell的情况只能出现在右侧、下侧、右下角
    // 这三种情况下的有效子cell依然有相邻关系。
    constexpr int offset[4][2] = {
        { 0, -1}, // dx, dy
        { 1,  0},
        { 0,  1},
        {-1,  0},
    };
    auto cell_xy = xy_of_cell(cell);
    for (int t=std::size(offset); t>=0; --t){
      int y = cell_xy.y + offset[t][1],
          x = cell_xy.x + offset[t][0];
      if (y < 0 || y >= height || x < 0 || x >= width)
        continue;
      int j = idx_of_cell(y, x);
      const Cell &cell_j = cell_s[j];
      if (!cell_j.valid)
        continue;
      adj_cells.emplace(j);
    }
    return adj_cells;
  }
};

struct SuperPixel{
  const int id = -1;
  cv::Point2i xy_mean{-1, -1};
  cv::Vec3b rgb_mean{0, 0, 0};
  int n_cells = 0;

  SuperPixel(const Cell &cell)
   : id(cell.spx_id), n_cells(1) {
    assert(cell.valid);
    xy_mean.x = (cell.x_start + cell.x_end) / 2;
    xy_mean.y = (cell.y_start + cell.y_end) / 2;
    rgb_mean  = cell.rgb;
  }
};

inline
F32
dist_rgb(const cv::Vec3b &l, const cv::Vec3b &r){
  auto d = l - r;
  F32 dist = 0;
  for (int i=0; i<3; ++i)
    dist += (F32) d[i] * (F32) d[i];
  return dist;
}

inline
F32
dist_spatial(const cv::Point2i &l, const cv::Point2i &r){
  auto d = l - r;
  F32 dist = (F32) d.y * (F32) d.y + (F32) d.x * (F32) d.x;
  return dist;
}

cv::Mat_<int>
grid2label(const Grids &grids, int height, int width, bool label_spx=true){
  int n = 0;
  cv::Mat_<int> labels(height, width, -1);
  for(int j=grids.cell_s.size()-1; j>=0; --j){
    const auto &cell = grids.cell_s[j];
    if (!cell.valid)
      continue;
    for (int y=cell.y_start; y<=cell.y_end; ++y)
      for (int x=cell.x_start; x<=cell.x_end; ++x) {
        labels.at<int>(y, x) = label_spx ? cell.spx_id : j;
        ++n;
      }
  }
  assert(n == height * width);
  return labels;
}

inline
int
exp2(int v){
  if (v <= 0)
    return 1;
  if (v % 2 == 0){
    int r = exp2(v / 2);
    return r * r;
  } else {
    int r = exp2(v - 1);
    return 2 * r;
  }
}

void
split_grids(Grids &grids,
            std::vector<SuperPixel> &spx_s,
            const cv::Mat_<cv::Vec3b> &rgb_img){
  --grids.level;
  assert(grids.level >= 0);
  grids.cell_size = exp2(grids.level);
  assert(grids.cell_size % 2 == 0 || grids.cell_size == 1);
  grids.height *= 2;
  grids.width  *= 2;

#if 0
#define cell_range(cell) (cell).x_start, (cell).x_end, (cell).y_start, (cell).y_end
#endif

  std::vector<Cell> new_cell_s;
  new_cell_s.resize(grids.cell_s.size() * 4);
  for (int j=grids.cell_s.size()-1; j>=0; --j){
    const auto &cell = grids.cell_s[j];
    if (!cell.valid)
      continue;
    const int jy = j / (grids.width / 2), jx = j % (grids.width / 2);
    const int j0y = 2 * jy , j0x = 2 * jx ,
              j1y = j0y    , j1x = j0x + 1,
              j2y = j0y + 1, j2x = j0x    ,
              j3y = j0y + 1, j3x = j0x + 1;
    const int j0 = grids.idx_of_cell(j0y, j0x),
              j1 = grids.idx_of_cell(j1y, j1x),
              j2 = grids.idx_of_cell(j2y, j2x),
              j3 = grids.idx_of_cell(j3y, j3x);
    Cell &c0 = new_cell_s[j0],
         &c1 = new_cell_s[j1],
         &c2 = new_cell_s[j2],
         &c3 = new_cell_s[j3];
    assert(!c0.valid && !c1.valid && !c2.valid && !c3.valid);
    const int h = cell.y_end - cell.y_start + 1,
              w = cell.x_end - cell.x_start + 1;
    assert(h >= 1 && w >= 1);
    if (h > 1 && w > 1){
      c0.valid = c1.valid = c2.valid = c3.valid = true;
      c0.spx_id   = c1.spx_id   = c2.spx_id   = c3.spx_id   = cell.spx_id;
      c0.y_start  = cell.y_start;
      c0.y_end    = cell.y_start + (h-1)/2;
      c0.x_start  = cell.x_start;
      c0.x_end    = cell.x_start + (w-1)/2;
      c1.y_start  = c0.y_start;
      c1.y_end    = c0.y_end;
      c1.x_start  = c0.x_end + 1;
      c1.x_end    = cell.x_end;
      c2.y_start  = c0.y_end + 1;
      c2.y_end    = cell.y_end;
      c2.x_start  = c0.x_start;
      c2.x_end    = c0.x_end;
      c3.y_start  = c2.y_start;
      c3.y_end    = c2.y_end;
      c3.x_start  = c2.x_end + 1;
      c3.x_end    = cell.x_end;
      c0.rgb      = c0.mean_rgb(rgb_img);
      c1.rgb      = c1.mean_rgb(rgb_img);
      c2.rgb      = c2.mean_rgb(rgb_img);
      c3.rgb      = c3.mean_rgb(rgb_img);
      assert(cell.area() == c0.area() + c1.area() + c2.area() + c3.area());
#if 0
      printf("block(%d:%d,%d:%d) => block(%d:%d,%d:%d),"
             " block(%d:%d,%d:%d), block(%d:%d,%d:%d), block(%d:%d,%d:%d)\n",
             cell_range(cell), cell_range(c0), cell_range(c1), cell_range(c2), cell_range(c3));
      fflush(stdout);
#endif
      spx_s[cell.spx_id].n_cells += 3;//一拆为四
    } else if (h > 1 && w == 1) {
      c1.valid = c3.valid = false;
      c0.valid = c2.valid = true;
      c0.spx_id   = c2.spx_id   = cell.spx_id;
      c0.y_start  = cell.y_start;
      c0.y_end    = cell.y_start + (h-1)/2;
      c0.x_start  = cell.x_start;
      c0.x_end    = cell.x_end;
      c2.y_start  = c0.y_end + 1;
      c2.y_end    = cell.y_end;
      c2.x_start  = cell.x_start;
      c2.x_end    = cell.x_end;
      c0.rgb      = c0.mean_rgb(rgb_img);
      c2.rgb      = c2.mean_rgb(rgb_img);
      assert(cell.area() == c0.area() + c2.area());
#if 0
      printf("block(%d:%d,%d:%d) => block(%d:%d,%d:%d) and block(%d:%d,%d:%d)\n",
             cell_range(cell), cell_range(c0), cell_range(c2));
      fflush(stdout);
#endif
      spx_s[cell.spx_id].n_cells += 1;//一拆为二
    } else if (h == 1 && w > 1) {
      c2.valid = c3.valid = false;
      c0.valid = c1.valid = true;
      c0.spx_id   = c1.spx_id   = cell.spx_id;
      c0.y_start  = cell.y_start;
      c0.y_end    = cell.y_end;
      c0.x_start  = cell.x_start;
      c0.x_end    = cell.x_start + (w-1)/2;
      c1.y_start  = cell.y_start;
      c1.y_end    = cell.y_end;
      c1.x_start  = c0.x_end + 1;
      c1.x_end    = cell.x_end;
      c0.rgb      = c0.mean_rgb(rgb_img);
      c1.rgb      = c1.mean_rgb(rgb_img);
      assert(cell.area() == c0.area() + c1.area());
#if 0
      printf("block(%d:%d,%d:%d) => block(%d:%d,%d:%d) and block(%d:%d,%d:%d)\n",
             cell_range(cell), cell_range(c0), cell_range(c1));
      fflush(stdout);
#endif
      spx_s[cell.spx_id].n_cells += 1;//一拆为二
    } else if (h == 1 && w == 1) {
      c1.valid = c2.valid = c3.valid = false;
      c0.valid = true;
      c0.spx_id   = cell.spx_id;
      c0.y_start  = cell.y_start;
      c0.y_end    = cell.y_end;
      c0.x_start  = cell.x_start;
      c0.x_end    = cell.x_end;
      c0.rgb      = c0.mean_rgb(rgb_img);
      assert(cell.area() == c0.area());
#if 0
      printf("block(%d:%d,%d:%d) => block(%d:%d,%d:%d)\n",
             cell_range(cell), cell_range(c0));
      fflush(stdout);
#endif
      //spx_s[cell.spx_id].n_cells += 0;//不变
    } else {
      assert(0 && "impossible!");
    }
  }
  grids.cell_s.swap(new_cell_s);
  assert(grids.cell_s.size() == grids.height * grids.width);
#if 0
  for (const auto &cell : grids.cell_s){
    printf("block(%d:%d, %d:%d)\n",cell_range(cell));
  }
  fflush(stdout);
  auto labels = grid2label(grids, rgb_img.rows, rgb_img.cols, false);
  save_edge_map("t.png", labels);
  save_segmentation_map("t2.png", labels);
#endif
}

void
initialize(Grids &grids,
           std::vector<SuperPixel> &spx_s,
           const cv::Mat_<cv::Vec3b> &rgb_img,
           const HyperParams &params){
  const int n_pixel = rgb_img.rows * rgb_img.cols;
  const F32 square = std::sqrt((float)n_pixel / params.expect_spx_num);
  //1,2,4,8,..,2^m,2^(m+1)
  const int m = (F32) std::log2(square);
  const int cell_size = exp2(m);
  const int grid_height = ((rgb_img.rows + cell_size - 1) / cell_size),
            grid_width  = ((rgb_img.cols + cell_size - 1) / cell_size);
  const int n_spx = grid_height * grid_width;
  spx_s.clear();
  spx_s.reserve(n_spx);
  grids.level     = m;//invalid level, need to split.
  grids.cell_size = cell_size;
  grids.cell_s.clear();
  grids.cell_s.reserve(n_spx);
  grids.height    = grid_height;
  grids.width     = grid_width;
  for (int y_start=0; y_start < rgb_img.rows; y_start+=cell_size) {
    int y_end = std::min(y_start + cell_size, rgb_img.rows) - 1;
    for (int x_start = 0; x_start < rgb_img.cols; x_start += cell_size) {
      int x_end = std::min(x_start + cell_size, rgb_img.cols) - 1;
      const int spx_id = spx_s.size();
      grids.cell_s.emplace_back(spx_id,
                                x_start, x_end, y_start, y_end,
                                true,
                                rgb_img);
      spx_s.emplace_back(grids.cell_s.back());
    }
  }
  assert(grids.cell_s.size() == n_spx && spx_s.size() == n_spx);

  for (int j=(grids.cell_s.size()-1); j>=0; --j) {
    const Cell &cell_j = grids.cell_s[j];
    cv::Point2i cell_j_xy((cell_j.x_start + cell_j.x_end) / 2,
                          (cell_j.y_start + cell_j.y_end) / 2);
    for (int k = j - 1; k >= 0; --k) {
      const Cell &cell_k = grids.cell_s[k];
      cv::Point2i cell_k_xy((cell_k.x_start + cell_k.x_end)/2,
                            (cell_k.y_start + cell_k.y_end)/2);
      F32 d_rgb = dist_rgb(cell_j.rgb, cell_k.rgb);
      F32 d_spatial = dist_spatial(cell_j_xy, cell_k_xy);
      if (d_rgb > params._record_max_rgb_dist)
        params._record_max_rgb_dist = d_rgb;
      if (d_spatial > params._record_max_spatial_dist)
        params._record_max_spatial_dist = d_spatial;
    }
  }
}

std::vector<int>
find_boundary_cells(const Grids &grids){
  std::vector<int> boundary_cells;
  assert(grids.cell_s.size() == grids.height * grids.width);
  for (int y=0; y<grids.height; ++y)
    for (int x=0; x<grids.width; ++x){
      int i  = grids.idx_of_cell(y, x);
      const Cell &cell_i = grids.cell_s[i];
      if (!cell_i.valid)
        continue;

      bool on_border = false;
      auto adj_cells = grids.adjacent_cells(i);
      for (int j : adj_cells) {
        const Cell &cell_j = grids.cell_s[j];
        assert(cell_j.valid);
        if (cell_j.spx_id != cell_i.spx_id){
          on_border = true;
          break;
        }
      }
      //Future: 排除那些使spx一分为二的block
      if (on_border)
        boundary_cells.emplace_back(i);
    }
  return boundary_cells;
}

bool
break_if_remove(int cell, const Grids &grids){
  //todo
  return false;
}

F32
cell_energy_item(const Cell &cell,
                 const SuperPixel &spx,
                 const HyperParams &params){
  assert(cell.valid);
  assert(spx.n_cells > 0);
  F32 d_rgb = dist_rgb(cell.rgb, spx.rgb_mean);
  if (d_rgb > params._record_max_rgb_dist)
    params._record_max_rgb_dist = d_rgb;
  d_rgb /= params._max_rgb_dist;

  cv::Point2i cell_xy((cell.x_start + cell.x_end)/2, (cell.y_start + cell.y_end)/2);
  F32 d_spatial = dist_spatial(cell_xy, spx.xy_mean);
  if (d_spatial > params._record_max_spatial_dist)
    params._record_max_spatial_dist = d_spatial;
  d_spatial /= params._max_spatial_dist;

  F32 d = d_rgb + F32(params.spatial_scale) * d_spatial;//fixme other dist item
  return d;
}

int
arg_spx_min_energy(int cell,
                   const Grids &grids,
                   const std::vector<SuperPixel> &spx_s,
                   const HyperParams &params){
  const int cell_spx_id = grids.cell_s[cell].spx_id;
  auto adj_cells = grids.adjacent_cells(cell);
  std::vector<int> adj_spx;
  for (int i : adj_cells) {
    const int spx_id = grids.cell_s[i].spx_id;
    if (spx_id != cell_spx_id)
      adj_spx.emplace_back(spx_id);
  }

  const Cell &target_cell = grids.cell_s[cell];
  int best_spx_id = target_cell.spx_id;
  F32 best_energy = cell_energy_item(target_cell, spx_s[best_spx_id], params);
  //如果调整target_cell为其他spx，将降低总能量，则调整之
  //todo 当前的能量函数设计很差
  //     cell 有三种情况：不变，邻，新
  for (int spx_id : adj_spx){
    F32 new_energy = cell_energy_item(target_cell, spx_s[spx_id], params);
    if (new_energy < best_energy){
      best_energy = new_energy;
      best_spx_id = spx_id;
    }
  }

  return best_spx_id;
}

void
update_cell_and_spx(int cell,
                    int spx_id,
                    Grids &grids,
                    std::vector<SuperPixel> &spx_s){
  Cell &target_cell  = grids.cell_s[cell];
  assert(target_cell.valid);
  SuperPixel &old_spx = spx_s[target_cell.spx_id];
  SuperPixel &new_spx = spx_s[spx_id];
  assert(old_spx.n_cells >= 0);
  assert(new_spx.n_cells >= 0);

  target_cell.spx_id = spx_id;

  F64 x_sum = (F64)old_spx.xy_mean.x * (F64)old_spx.n_cells - F64((target_cell.x_start + target_cell.x_end) / 2),
      y_sum = (F64)old_spx.xy_mean.y * (F64)old_spx.n_cells - F64((target_cell.y_start + target_cell.y_end) / 2);
  old_spx.xy_mean.x = (x_sum / (F64) std::max(1, old_spx.n_cells - 1));
  old_spx.xy_mean.y = (y_sum / (F64) std::max(1, old_spx.n_cells - 1));
  F64 r_sum = (F64)old_spx.rgb_mean[0] * (F64)old_spx.n_cells - F64(target_cell.rgb[0]),
      g_sum = (F64)old_spx.rgb_mean[1] * (F64)old_spx.n_cells - F64(target_cell.rgb[1]),
      b_sum = (F64)old_spx.rgb_mean[2] * (F64)old_spx.n_cells - F64(target_cell.rgb[2]);
  old_spx.rgb_mean[0] = clamp2uchar(r_sum / (F64) std::max(1, old_spx.n_cells - 1));
  old_spx.rgb_mean[1] = clamp2uchar(g_sum / (F64) std::max(1, old_spx.n_cells - 1));
  old_spx.rgb_mean[2] = clamp2uchar(b_sum / (F64) std::max(1, old_spx.n_cells - 1));
  --old_spx.n_cells;

  x_sum = (F64)new_spx.xy_mean.x * (F64)new_spx.n_cells + F64((target_cell.x_start + target_cell.x_end) / 2);
  y_sum = (F64)new_spx.xy_mean.y * (F64)new_spx.n_cells + F64((target_cell.y_start + target_cell.y_end) / 2);
  new_spx.xy_mean.x = (x_sum / F64(new_spx.n_cells + 1));
  new_spx.xy_mean.y = (y_sum / F64(new_spx.n_cells + 1));
  r_sum = (F64)new_spx.rgb_mean[0] * (F64)new_spx.n_cells + F64(target_cell.rgb[0]),
  g_sum = (F64)new_spx.rgb_mean[1] * (F64)new_spx.n_cells + F64(target_cell.rgb[1]),
  b_sum = (F64)new_spx.rgb_mean[2] * (F64)new_spx.n_cells + F64(target_cell.rgb[2]);
  new_spx.rgb_mean[0] = clamp2uchar(r_sum / F64(new_spx.n_cells + 1));
  new_spx.rgb_mean[1] = clamp2uchar(g_sum / F64(new_spx.n_cells + 1));
  new_spx.rgb_mean[2] = clamp2uchar(b_sum / F64(new_spx.n_cells + 1));
  ++new_spx.n_cells;
}


cv::Mat_<int>
etps_(const cv::Mat_<cv::Vec3b> &rgb_img,
     const HyperParams &params){
  cv::Mat_<cv::Vec3b> img;
  if (params.rgb2lab) {
    img = rgb2lab(rgb_img);
    if (params.verbose)
      save_image("input_image_lab.png", img);
  } else {
    img = rgb_img;
  }

  Grids grids;
  std::vector<SuperPixel> spx_s;
  initialize(grids, spx_s, img, params);
  params.update_record();
  if (params.verbose) {
    auto labels = grid2label(grids, rgb_img.rows, rgb_img.cols);
    save_edge_map(
        str_printf(512, "%s_init_mat.png", __func__),
        labels);
    save_segmentation_map(
        str_printf(512, "%s_init_mat_color.png", __func__),
        labels);
    std::vector<cv::Point2i> spx_pos;
    spx_pos.reserve(spx_s.size());
    for (int j=spx_s.size()-1; j>=0; --j)
      if (spx_s[j].n_cells > 0)
        spx_pos.emplace_back(spx_s[j].xy_mean);
    save_super_pixel(
        str_printf(512, "%s_init_super_pixel.png", __func__),
        spx_pos, labels, rgb_img);
  }

  while (grids.level > 0){
    split_grids(grids, spx_s, img);
    assert(grids.level >= 0);
    if (params.verbose){
      printf("start at level-%d(decreasing)...\n", grids.level);
      fflush(stdout);
    }

    for (int iter=params.max_iter_num_in_each_level; iter>0; --iter) {
      if (params.verbose){
        printf("at iterations-%d(decreasing)...\n", iter);
        fflush(stdout);
      }
      std::vector<int> boundary_cells = find_boundary_cells(grids);
      std::vector<uchar> in_boundary_cells;
      in_boundary_cells.resize(grids.cell_s.size(), 0);
      for (int c : boundary_cells)
        in_boundary_cells[c] = 1;

      while (!boundary_cells.empty()) {
        int cell = boundary_cells.back();
        boundary_cells.pop_back();
        if (break_if_remove(cell, grids))
          continue;

        int spx_id = arg_spx_min_energy(cell, grids, spx_s, params);
        if (spx_id == grids.cell_s[cell].spx_id)
          continue;
        update_cell_and_spx(cell, spx_id, grids, spx_s);

        auto adjacent_cells = grids.adjacent_cells(cell);
        for (int c : adjacent_cells) {
          if (in_boundary_cells[c])
            continue;
          boundary_cells.push_back(c);
          in_boundary_cells[c] = 1;
        }
      }

      params.update_record();
      if (params.verbose) {
        auto labels = grid2label(grids, rgb_img.rows, rgb_img.cols);
        save_edge_map(
            str_printf(512, "%s_mat_%d-%d.png", __func__, grids.level, iter),
            labels);
        save_segmentation_map(
            str_printf(512, "%s_mat_%d-%d_color.png", __func__, grids.level, iter),
            labels);
        std::vector<cv::Point2i> spx_pos;
        spx_pos.reserve(spx_s.size());
        for (int j=spx_s.size()-1; j>=0; --j)
          if (spx_s[j].n_cells > 0)
            spx_pos.emplace_back(spx_s[j].xy_mean);
        save_super_pixel(
            str_printf(512, "%s_img_%d-%d.png", __func__, grids.level, iter),
            spx_pos, labels, rgb_img);
      }
    }
  }

  assert(grids.cell_size == 1);
  assert(grids.cell_s.size() >= rgb_img.rows * rgb_img.cols);
  auto labels = grid2label(grids, rgb_img.rows, rgb_img.cols);
  return labels;
}

cv::Mat_<int>
etps(const cv::Mat_<cv::Vec3b> &rgb_img,
     const HyperParams &params) {
  cv::Mat_<int> result;
  try {
    result = etps_(rgb_img, params);
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

