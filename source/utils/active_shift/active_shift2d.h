/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_ASL_OPS_H_
#define TENSORFLOW_KERNELS_ASL_OPS_H_

// #define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <cstring>
#include <vector>

namespace tensorflow {
// typedef Eigen::ThreadPoolDevice CPUDevice;
// typedef std::vector<int> TShape;
// typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename DType>
struct ASLForward{
	void operator()(const Device& d,
			const int count, const int num, const int channels,
			const int top_height, const int top_width,
			const int bottom_height, const int bottom_width,
			const DType* xshift, const DType* yshift,
			const int pad_h, const int pad_w,
			const int stride_h, const int stride_w,
			const DType* bottom_data, DType* top_data);
};


template <typename Device, typename DType>
struct ASL_ShiftBackward{
	void operator()(const Device& d,
			const int count, const int num, const int channels,
			const int top_height, const int top_width,
			const int bottom_height, const int bottom_width,
			const DType* xshift, const DType* yshift,
			const int pad_h, const int pad_w,
			const int stride_h, const int stride_w,
			const DType* bottom_data, const DType* top_diff,
			DType* bottom_backprop_ptr, DType* offset_backprop_ptr, DType* temp_buf_ptr);
};

template <typename Device, typename DType>
struct ASL_BottomBackward{
	void operator()(const Device& d,
			const int count, const int num, const int channels,
			const int top_height, const int top_width,
			const int bottom_height, const int bottom_width,
			const DType* xshift, const DType* yshift,
			const int pad_h, const int pad_w,
			const int stride_h, const int stride_w,
			const DType* bottom_data, const DType* top_diff,
			DType* bottom_backprop_ptr, DType* offset_backprop_ptr);
};

template <typename Device, typename DType>
struct ApplyShiftConstraint{
	void operator()(const Device& d, const int count,
			const DType* xshift_data, const DType* yshift_data,
			DType* xshift_diff, DType* yshift_diff,
			const bool normalize, const float clip_gradient);
};

template <typename Device, typename DType>
struct setZero {
    void operator() (const Device& d, const int n, DType* out);
};

template <typename Device, typename DType>
struct setValue {
    void operator() (const Device& d, const int n, const DType val, DType* out);
};


}  // namespace functor
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_ASL_OPS_H_
