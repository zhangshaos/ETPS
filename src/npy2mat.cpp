// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "npy2mat.h"
#include <string>
#include <fstream>
#include <opencv2/core.hpp>


namespace cvDNN {

/*
 * Following codes come from:
 * https://github.com/opencv/opencv/blob/master/modules/dnn/test/npy_blob.cpp
 */

std::string
getType(const std::string &header) {
  std::string field = "'descr':";
  size_t idx = header.find(field);
  CV_Assert(idx != -1);

  size_t from = header.find('\'', idx + field.size()) + 1;
  size_t to = header.find('\'', from);
  return header.substr(from, to - from);
}

std::string
getFortranOrder(const std::string &header) {
  std::string field = "'fortran_order':";
  size_t idx = header.find(field);
  CV_Assert(idx != -1);

  size_t from = header.find_last_of(' ', idx + field.size()) + 1;
  size_t to = header.find(',', from);
  return header.substr(from, to - from);
}

std::vector<int>
getShape(const std::string &header) {
  std::string field = "'shape':";
  size_t idx = header.find(field);
  CV_Assert(idx != -1);

  size_t from = header.find('(', idx + field.size()) + 1;
  size_t to = header.find(')', from);

  std::string shapeStr = header.substr(from, to - from);
  if (shapeStr.empty())
    return std::vector<int>(1, 1);

  // Remove all commas.
  shapeStr.erase(std::remove(shapeStr.begin(), shapeStr.end(), ','),
                 shapeStr.end());

  std::istringstream ss(shapeStr);
  int value;

  std::vector<int> shape;
  while (ss >> value) {
    shape.push_back(value);
  }
  return shape;
}

/**
 * 从.npy文件中加载cv:Mat对象，目前仅支持CV_32F,CV_32S,CV_8U的数据格式
 * @param path
 * @param rtype
 * @return
 */
cv::Mat
blobFromNPY(const std::string &path, int rtype) {
  std::ifstream ifs(path.c_str(), std::ios::binary);
  CV_Assert(ifs.is_open());

  std::string magic(6, '*');
  ifs.read(&magic[0], magic.size());
  CV_Assert(magic == "\x93NUMPY");

  ifs.ignore(1);  // Skip major version byte.
  ifs.ignore(1);  // Skip minor version byte.

  unsigned short headerSize;
  ifs.read((char *) &headerSize, sizeof(headerSize));

  std::string header(headerSize, '*');
  ifs.read(&header[0], header.size());

  // Extract data type.
  std::string sType;
  if (rtype == CV_32S)
    sType = "<i4";
  else if (rtype == CV_32F)
    sType = "<f4";
  else if (rtype == CV_8U)
    sType = "|u1";
  else
    CV_Assert("Not support now!");
  CV_Assert(getType(header) == sType);
  CV_Assert(getFortranOrder(header) == "False");
  std::vector<int> shape = getShape(header);

  cv::Mat blob(shape, rtype);
  ifs.read((char *) blob.data, blob.total() * blob.elemSize());
  CV_Assert((size_t) ifs.gcount() == blob.total() * blob.elemSize());

  return blob;
}

}
