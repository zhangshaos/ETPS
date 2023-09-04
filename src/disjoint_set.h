#ifndef DISJOINT_SET_H
#define DISJOINT_SET_H

#include <cstdint>
#include <vector>


//（连续uint32编号的）并查集
struct DisjointSet {
  std::vector<uint32_t> parent;//parent[i] 表示类别i的父类别

  bool
  contain_class(uint32_t c)const;
  uint32_t
  find_root_class(uint32_t c)const;
  bool
  is_root_class(uint32_t c)const;
  bool
  try_add_class(uint32_t c);
  void
  union_class(uint32_t c1, uint32_t c2);
  uint32_t
  shrink_parent(std::vector<uint32_t> *out_shrink_parent = nullptr);
};


#endif //DISJOINT_SET_H
