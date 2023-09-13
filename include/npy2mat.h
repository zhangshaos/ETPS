#ifndef ELSED_NPY2CVMAT_H
#define ELSED_NPY2CVMAT_H

#include <string>
#include <opencv2/core.hpp>

namespace cvDNN {

/**
 * 从.npy文件中加载cv:Mat对象，目前仅支持CV_32F,CV_32S,CV_8U的数据格式
 * @param path
 * @param rtype
 * @return
 */
cv::Mat
blobFromNPY(const std::string &path, int rtype=CV_32F);

}

#endif //ELSED_NPY2CVMAT_H
