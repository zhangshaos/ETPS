# ETPS

## master branch

Implement of the paper "Real-Time Coarse-to-fine Topologically Preserving Segmentation".

## fh branch

Implement of the paper "Efficient graph-based image segmentation".

## CCS branch

Implement of the paper "Convexity constrained efficient superpixel and supervoxel extraction".


## CCS_py branch

在 CCS 方法的基础上优化语义分割结果。

### 编译

0. 此仓库依赖 OpenCV

1. git clone 此仓库后，依次运行：
```shell
mkdir build && cd build && cmake ..
cmake --build
```

2. 正常编译完成后，产生如下结果：

`lib/libCCS.a` 静态库文件  
`bin/pyCCS.__PYTHON_VERSION__.so` 提供给Python调用的库文件  
`bin/libcpptrace.so` 当前库依赖的动态库文件  
`bin/demo` 一个简单的C++使用当前库的案例，编译完成后，可以直接运行此文件   
`bin/demo_sem_seg` 一个复杂的C++使用当前库的案例，需要配置，不可直接运行，下文介绍。


### demo

在 demo/ 目录下展示了使用C++/Python使用此库的案例。

- CMakeLists.txt 展示如果将当前静态库连接到可执行文件中；
- main.cpp `bin/demo`的源文件，一个可直接运行的简单例子；
- sem_seg_main.cpp `bin/demo_sem_seg`的源文件，需要配置如下项：
    - img_dir 输入图片目录（图片后缀为.JPG）
    - label_dir 图片语义分割logits得分（一个(H,W,C)的矩阵a，a[y,x,c]表示像素(y,x)数据c类的概率值，该值在0~1之间，文件后缀为.npy）
    - names[] 输入图片的文件名（不包含文件扩展名.JPG）    
      注意，img_dir和label_dir目录下的文件名（不包含扩展名）必须一致。

- t.py 一个可直接运行的简单例子，`bin/demo`的Python版本；
- t2.py 一个需要和sem_seg_main.cpp同样配置的例子，`bin/deo_sem_seg`的Python版本。

注意：
- c++调用者在引用此库时，需要链接到 CCS 和 OpenCV 两个库
- python调用者在引用此库时，注意将`bin/pyCCS.__PYTHON_VERSION__.so`所在目录加入到`sys.path`搜索路径中，并`import pyCCS`库


### API

python: `ccs(rgb_img: np.ndarray[np.uint8], expect_spx_num=1000, spatial_scale=0.3f, rgb2lab=true, verbose=True) -> np.ndarray[np.int32]`  

c++: `ccs_segment(rgb_img: cv::Mat_<cv::Vec3b>, expect_spx_num=1000, spatial_scale=0.3f, rgb2lab=true, verbose=True) -> cv::Mat_<int>`  

- rgb_img: RGB格式输入图片；
- expect_spx_num: 预期的超像素数量，一般情况下，结果中超像素数量都大于这个数；
- spatial_scale: 空间距离的缩放因子，请参考 Convexity constrained efficient
  Superpixel and supervoxel extraction, SPIC 2015.  
  此项越大，超像素形状越规则，反之，形状越接近颜色边界；
- rgb2lab: 使用LAB颜色空间而不是RGB颜色空间；
- verbose: 算法运行过程中输出日志和中间结果；
- 返回值：(H,W)矩阵a，a[y,x]表示像素(y,x)所属的超像素编号。

---

python: `naive_segment(spx_label: np.ndarray[np.int32], rgb_image: np.ndarray[np.uint8], semantic_logits: np.ndarray[np.float32]) -> np.ndarray[np.int32]`  

c++: `naive_semantic_segment(spx_label: cv::Mat_<int>, rgb_image: cv::Mat_<cv::Vec3b>, semantic_logits: cv::Mat_<float>) -> cv::Mat_<int>`  

- spx_label: (H,W)矩阵a，a[y,x]表示像素(y,x)所属的超像素编号；
- rgb_image: RGB格式的输入图片；
- semantic_logits: (H,W,C)矩阵b，b[y,x,c]表示像素(y,x)属于类别c的概率值；
- 返回值：(H,W)矩阵a，a[y,x]表示像素(y,x)所属的类别编号。

---

python: `crf_segment(spx_label: np.ndarray[np.int32], rgb_image: np.ndarray[np.uint8], semantic_logits: np.ndarray[np.float32], wi=10, wj=1, verbose=True) -> np.ndarray[np.int32]`  

c++: `crf_semantic_segment(spx_label: cv::Mat_<int>, rgb_image: cv::Mat_<cv::Vec3b>, semantic_logits: cv::Mat_<float>, wi=10, wj=1, verbose=True) -> cv::Mat_<int>`  

- spx_label: (H,W)矩阵a，a[y,x]表示像素(y,x)所属的超像素编号；
- rgb_image: RGB格式的输入图片；
- semantic_logits: (H,W,C)矩阵b，b[y,x,c]表示像素(y,x)属于类别c的概率值；
- wi: CRF能量方程中，势函数φ(xi)的系数；
- wj: CRF能量方程中，势函数φ(xi,xj)的系数；
- verbose: 算法运行过程中输出日志和中间结果；
- 返回值：(H,W)矩阵a，a[y,x]表示像素(y,x)所属的类别编号。

---

python: `mrf_segment(spx_label: np.ndarray[np.int32], rgb_image: np.ndarray[np.uint8], semantic_logits: np.ndarray[np.float32], wi=10, wj=1, verbose=True) -> np.ndarray[np.int32]`  

c++: `mrf_semantic_segment(spx_label: cv::Mat_<int>, rgb_image: cv::Mat_<cv::Vec3b>, semantic_logits: cv::Mat_<float>, wi=10, wj=1, verbose=True) -> cv::Mat_<int>`  

- spx_label: (H,W)矩阵a，a[y,x]表示像素(y,x)所属的超像素编号；
- rgb_image: RGB格式的输入图片；
- semantic_logits: (H,W,C)矩阵b，b[y,x,c]表示像素(y,x)属于类别c的概率值；
- wi: MRF能量方程中，势函数φ(xi)的系数；  
- wj: MRF能量方程中，势函数φ(xi,xj)的系数；   
- verbose: 算法运行过程中输出日志和中间结果；
- 返回值：(H,W)矩阵a，a[y,x]表示像素(y,x)所属的类别编号。
