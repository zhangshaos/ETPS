#include "disjoint_set.h"
#include <cstdint>
#include <vector>
#include <unordered_map>


bool
DisjointSet::contain_class(uint32_t c) const {
  return parent.size() > c && parent[c] != uint32_t(-1);
}

uint32_t
DisjointSet::find_root_class(uint32_t c) const {
  if (contain_class(c))
    while (c != parent[c])
      c = parent[c];
  else
    c = (uint32_t) -1;
  return c;
}

bool
DisjointSet::is_root_class(uint32_t c) const {
  return contain_class(c) && find_root_class(c) == c;
}

bool
DisjointSet::try_add_class(uint32_t c) {
  if (c >= parent.size())
    parent.resize(c + 1, uint32_t(-1));
  if (!contain_class(c)) {
    parent[c] = c;
    return true;
  } else
    // this ID is existed.
    return false;
}

void
DisjointSet::union_class(uint32_t c1, uint32_t c2) {
  try_add_class(c1);
  try_add_class(c2);
  c1 = find_root_class(c1);
  c2 = find_root_class(c2);
  if (c1 <= c2)
    parent[c2] = c1;
  else
    parent[c1] = c2;
}

uint32_t
DisjointSet::shrink_parent(std::vector<uint32_t> *out_shrink_parent) {
  //合并类别中间的空白。
  // 在所有类型合并结束后，每个类别的根类别可能出现[2,2,2,0,0,5,5,5]这种情况，
  // 将其收缩为 [0,0,0,1,1,2,2,2]
  std::vector<uint32_t> shrink_parent(parent.size(), (uint32_t) -1);
  uint32_t start_parent_class = 0;
  std::unordered_map<uint32_t, uint32_t> root2shrink_cls;
  root2shrink_cls.reserve(parent.size());
  for (uint32_t i = 0, i_end = (uint32_t) parent.size(); i < i_end; ++i) {
    if (!contain_class(i))
      //并查集中没有类别i
      continue;
    uint32_t root_cls = find_root_class(i);
    parent[i] = root_cls;
    if (root2shrink_cls.count(root_cls)) {
      shrink_parent[i] = root2shrink_cls.at(root_cls);
    } else {
      root2shrink_cls[root_cls] = start_parent_class;
      shrink_parent[i] = start_parent_class;
      ++start_parent_class;
    }
  }
  if (out_shrink_parent)
    out_shrink_parent->swap(shrink_parent);
  return start_parent_class;
}