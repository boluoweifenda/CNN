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

// #if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "active_shift2d.h"
#include "cuda.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
//#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include <algorithm>
#include <cstring>
#include <vector>


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
typedef std::vector<int32> TShape;

template <typename DType>
__global__ void ASLForward_GPUKernel(const int count, const int num, const int channels,
		const int top_height, const int top_width,
		const int bottom_height, const int bottom_width,
		const DType* xshift, const DType* yshift,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const DType* bottom_data, DType* top) {
	CUDA_1D_KERNEL_LOOP(index, count)
	{
		//n,c,i
		const int top_sp_dim = top_height * top_width;
		const int bottom_sp_dim = bottom_height * bottom_width;
		const int n = index/(channels * top_sp_dim);
		const int idx = index%(channels * top_sp_dim);
		const int c = idx/top_sp_dim;
		const int sp_idx = idx%top_sp_dim;
		const int h = sp_idx/top_width;
		const int w = sp_idx%top_width;
		const DType* data_im_ptr = bottom_data + n*channels*bottom_sp_dim + c*bottom_sp_dim;

		const int h_offset = h * stride_h - pad_h;
		const int w_offset = w * stride_w - pad_w;

		{
			DType val = 0;
			const DType x = xshift[c];
			const DType y = yshift[c];

			int h_im, w_im;

			int x1 = floorf(x);
			int x2 = x1+1;
			int y1 = floorf(y);
			int y2 = y1+1;

			h_im = h_offset + y1;
			w_im = w_offset + x1;
			DType q11 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

			h_im = h_offset + y1;
			w_im = w_offset + x2;
			DType q21 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

			h_im = h_offset + y2;
			w_im = w_offset + x1;
			DType q12 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

			h_im = h_offset + y2;
			w_im = w_offset + x2;
			DType q22 = (h_im >= 0 && w_im >= 0 && h_im < bottom_height && w_im < bottom_width) ? data_im_ptr[h_im*bottom_width + w_im] : 0;

			DType dx = x-x1;
			DType dy = y-y1;

			val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;

			top[index] = val;
		}
	}
}



template <typename DType>
__global__ void ASL_ShiftBackwardMerged_GPUKernel(const int count, const int num, const int channels,
		const int top_height, const int top_width,
		const int bottom_height, const int bottom_width,
		const DType* xshift, const DType* yshift,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const DType* bottom_data, const DType* top_diff, DType* shift_temp_buff_x, DType* shift_temp_buff_y) {
	CUDA_1D_KERNEL_LOOP(index, count)
	{
		//n,c,i
		const int top_sp_dim = top_height * top_width;
		const int bottom_sp_dim = bottom_height * bottom_width;
		const int n = index/(channels * top_sp_dim);
		const int idx = index%(channels * top_sp_dim);
		const int c = idx/top_sp_dim;
		const int sp_idx = idx%top_sp_dim;
		const int h = sp_idx/top_width;
		const int w = sp_idx%top_width;
		const DType* data_im_ptr = bottom_data + n*channels*bottom_sp_dim + c*bottom_sp_dim;

		const int h_offset = h * stride_h - pad_h;
		const int w_offset = w * stride_w - pad_w;

		//output : 2*(C) x (H*W)
		DType* out_ptr_x = shift_temp_buff_x + c*top_sp_dim + sp_idx; //(c,h,w)
		DType* out_ptr_y = shift_temp_buff_y + c*top_sp_dim + sp_idx; //(c,h,w)

		{
			DType val_x = 0, val_y = 0;

			const DType shiftX = xshift[c];
			const DType shiftY = yshift[c];


			//Calc
			const int ix1 = floorf(shiftX);
			const int ix2 = ix1+1;
			const int iy1 = floorf(shiftY);
			const int iy2 = iy1+1;
			const DType dx = shiftX-ix1;
			const DType dy = shiftY-iy1;

			const int h_im1 = h_offset + iy1;
			const int h_im2 = h_offset + iy2;
			const int h_im1a = h_im1 + ((dy==0)?-1:0);

			const int w_im1 = w_offset + ix1;
			const int w_im2 = w_offset + ix2;
			const int w_im1a = w_im1 + ((dx==0)?-1:0);

			const DType q11 = (h_im1 >= 0 && w_im1 >= 0 && h_im1 < bottom_height && w_im1 < bottom_width) ? data_im_ptr[h_im1*bottom_width + w_im1] : 0;
			const DType q21 = (h_im1 >= 0 && w_im2 >= 0 && h_im1 < bottom_height && w_im2 < bottom_width) ? data_im_ptr[h_im1*bottom_width + w_im2] : 0;
			const DType q12 = (h_im2 >= 0 && w_im1 >= 0 && h_im2 < bottom_height && w_im1 < bottom_width) ? data_im_ptr[h_im2*bottom_width + w_im1] : 0;
			const DType q22 = (h_im2 >= 0 && w_im2 >= 0 && h_im2 < bottom_height && w_im2 < bottom_width) ? data_im_ptr[h_im2*bottom_width + w_im2] : 0;


			const DType q11a = (dx==0 || dy==0)?((h_im1a >= 0 && w_im1a >= 0 && h_im1a < bottom_height && w_im1a < bottom_width) ? data_im_ptr[h_im1a*bottom_width + w_im1a] : 0):q11;
			const DType q21a = (dy==0)?((h_im1a >= 0 && w_im2 >= 0 && h_im1a < bottom_height && w_im2 < bottom_width) ? data_im_ptr[h_im1a*bottom_width + w_im2] : 0):q21;
			const DType q12a = (dx==0)?((h_im2 >= 0 && w_im1a >= 0 && h_im2 < bottom_height && w_im1a < bottom_width) ? data_im_ptr[h_im2*bottom_width + w_im1a] : 0):q12;

			val_x = (1-dy)*(q21a-q11a)+dy*(q22-q12a);
			val_y = (1-dx)*(q12a-q11a)+dx*(q22-q21a);


			//Summary
			const DType diff_scale = top_diff[index];
			const DType shiftX_diff_sum = val_x * diff_scale;;
			const DType shiftY_diff_sum = val_y * diff_scale;


			//reduce along batch dimension
			CudaAtomicAdd(out_ptr_x, shiftX_diff_sum);
			CudaAtomicAdd(out_ptr_y, shiftY_diff_sum);
		}
	}
}


template <typename DType>
__global__ void ASL_BottomBackward_Stride1_GPUKernel(const int count, const int channels,
		const int top_height, const int top_width, //top
		const int bottom_height, const int bottom_width, //bottom
		const DType* xshift, const DType* yshift,
		const int pad_h, const int pad_w,
		const DType* top_diff, DType* bottom_diff) {
	CUDA_1D_KERNEL_LOOP(index, count)
	{
		const int top_sp_dim = top_height * top_width;
		const int bottom_sp_dim = bottom_height * bottom_width;
		const int n = index/(channels * bottom_sp_dim);
		const int idx = index%(channels * bottom_sp_dim);
		const int c = idx/bottom_sp_dim;
		const int sp_idx = idx%bottom_sp_dim;
		const int h_col = sp_idx/bottom_width;
		const int w_col = sp_idx%bottom_width;
		const DType* top_diff_ptr = top_diff + n*channels*top_sp_dim + c*top_sp_dim;

		const int h_offset = h_col + pad_h;
		const int w_offset = w_col + pad_w;


		{
			DType val = 0;
			const DType x = -xshift[c];  //reverse shift
			const DType y = -yshift[c];

			int h_im, w_im;

			if(x==0 && y==0)
			{
				h_im = h_offset;
				w_im = w_offset;
				val = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
			}
			else
			{
				int x1 = floorf(x);
				int x2 = x1+1;
				int y1 = floorf(y);
				int y2 = y1+1;

				//q11
				DType q11 = 0;

				h_im = (h_offset + y1);
				w_im = (w_offset + x1);
				q11 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;

				//q21
				DType q21 = 0;

				h_im = (h_offset + y1);
				w_im = (w_offset + x2);
				q21 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;

				//q12
				DType q12 = 0;

				h_im = (h_offset + y2);
				w_im = (w_offset + x1);
				q12 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;

				//q22
				DType q22 = 0;

				h_im = (h_offset + y2);
				w_im = (w_offset + x2);
				q22 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;

				DType dx = x-x1;
				DType dy = y-y1;

				val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
				//printf("%d:%d, col(%d,%d), im(%d,%d) / %f,(%f,%f,%f,%f),(%f,%f)\n", flip, index, w_col, h_col, w_offset,h_offset, val, q11,q21,q12,q22, x,y);
			}

			bottom_diff[index] = val;
		}
	}
}


template <typename DType>
__global__ void ASL_BottomBackward_GPUKernel(const int count, const int channels,
		const int top_height, const int top_width, //top
		const int bottom_height, const int bottom_width, //bottom
		const DType* xshift, const DType* yshift,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const DType* top_diff, DType* bottom_diff) {
	CUDA_1D_KERNEL_LOOP(index, count)
	{
		const int top_sp_dim = top_height * top_width;
		const int bottom_sp_dim = bottom_height * bottom_width;
		const int n = index/(channels * bottom_sp_dim);
		const int idx = index%(channels * bottom_sp_dim);
		const int c = idx/bottom_sp_dim;
		const int sp_idx = idx%bottom_sp_dim;
		const int h_col = sp_idx/bottom_width;
		const int w_col = sp_idx%bottom_width;
		const DType* top_diff_ptr = top_diff + n*channels*top_sp_dim + c*top_sp_dim;

		const int h_offset = h_col + pad_h;
		const int w_offset = w_col + pad_w;


		{
			DType val = 0;
			const DType x = -xshift[c];  //reverse shift
			const DType y = -yshift[c];

			int h_im, w_im;

			if(x==0 && y==0)
			{
				h_im = h_offset;
				w_im = w_offset;
				if(w_im%stride_w == 0 && h_im%stride_h ==0)
				{
					w_im=w_im/stride_w;
					h_im=h_im/stride_h;

					val = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
				}
			}
			else
			{
				int x1 = floorf(x);
				int x2 = x1+1;
				int y1 = floorf(y);
				int y2 = y1+1;

				//q11
				DType q11 = 0;

				h_im = (h_offset + y1);
				w_im = (w_offset + x1);
				if(w_im%stride_w == 0 && h_im%stride_h ==0)
				{
					w_im=w_im/stride_w;
					h_im=h_im/stride_h;

					q11 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
				}

				//q21
				DType q21 = 0;

				h_im = (h_offset + y1);
				w_im = (w_offset + x2);
				if(w_im%stride_w == 0 && h_im%stride_h ==0)
				{
					w_im=w_im/stride_w;
					h_im=h_im/stride_h;

					q21 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
				}

				//q12
				DType q12 = 0;

				h_im = (h_offset + y2);
				w_im = (w_offset + x1);

				if(w_im%stride_w == 0 && h_im%stride_h ==0)
				{
					w_im=w_im/stride_w;
					h_im=h_im/stride_h;

					q12 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
				}

				//q22
				DType q22 = 0;

				h_im = (h_offset + y2);
				w_im = (w_offset + x2);

				if(w_im%stride_w == 0 && h_im%stride_h ==0)
				{
					w_im=w_im/stride_w;
					h_im=h_im/stride_h;

					q22 = (h_im >= 0 && w_im >= 0 && h_im < top_height && w_im < top_width) ? top_diff_ptr[h_im*top_width + w_im] : 0;
				}

				DType dx = x-x1;
				DType dy = y-y1;

				val = q11*(1-dx)*(1-dy) + q21*dx*(1-dy) + q12*(1-dx)*dy + q22*dx*dy;
				//printf("%d:%d, col(%d,%d), im(%d,%d) / %f,(%f,%f,%f,%f),(%f,%f)\n", flip, index, w_col, h_col, w_offset,h_offset, val, q11,q21,q12,q22, x,y);
			}

			bottom_diff[index] = val;
		}
	}
}


template <typename DType>
__global__ void ApplyShiftConstraint_GPUKernel(const int n, const DType* xshift_data, const DType* yshift_data, DType* xshift_diff, DType* yshift_diff, const bool normalize, const float clip_gradient) {
	CUDA_1D_KERNEL_LOOP(index, n) {
		const DType xi = xshift_data[index];
		const DType yi = yshift_data[index];
		const DType ri = sqrt(xi*xi+yi*yi);

		if(normalize) //normalize
		{
			const DType dx = xshift_diff[index];
			const DType dy = yshift_diff[index];
			const DType dr = sqrt(dx*dx+dy*dy);

			if(dr!=0)
			{
				xshift_diff[index] = dx/dr;
				yshift_diff[index] = dy/dr;
			}
		}
		else if(clip_gradient!=0)
		{
			const DType dx = xshift_diff[index];
			const DType dy = yshift_diff[index];
			const DType dr = sqrt(dx*dx+dy*dy);

			if(dr>clip_gradient)
			{
				xshift_diff[index] = dx/dr*clip_gradient;
				yshift_diff[index] = dy/dr*clip_gradient;
			}
		}
	}
}


template <typename DType>
__global__ void setValueKernel(const int n, const DType val, DType* result_data)
{

  CUDA_1D_KERNEL_LOOP(index, n) {
      *(result_data+index) = val;
  }
  
}


namespace functor {

template <typename DType>
struct ASLForward<GPUDevice, DType>{
	void operator()(const GPUDevice& d,
			const int count, const int num, const int channels,
			const int top_height, const int top_width,
			const int bottom_height, const int bottom_width,
			const DType* xshift, const DType* yshift,
			const int pad_h, const int pad_w,
			const int stride_h, const int stride_w,
			const DType* bottom_data, DType* top_data)
	{
		CudaLaunchConfig config = GetCudaLaunchConfig(count, d);

		ASLForward_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
		<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
				count, num, channels,
				top_height, top_width,
				bottom_height, bottom_width,
				xshift, yshift,
				pad_h, pad_w,
				stride_h, stride_w,
				bottom_data, top_data);
	}
};


template <typename DType>
struct ASL_ShiftBackward<GPUDevice, DType>{
	void operator()(const GPUDevice& d,
			const int count, const int num, const int channels,
			const int top_height, const int top_width,
			const int bottom_height, const int bottom_width,
			const DType* xshift, const DType* yshift,
			const int pad_h, const int pad_w,
			const int stride_h, const int stride_w,
			const DType* bottom_data, const DType* top_diff,
			DType* bottom_backprop_ptr, DType* offset_backprop_ptr, DType* temp_buf_ptr)
	{

		CudaLaunchConfig config = GetCudaLaunchConfig(count, d);

		const int temp_buf_offset = channels * top_height * top_width;	//(C*K) x (1*H*W)

		DType* shift_temp_buff_x = temp_buf_ptr;
		DType* shift_temp_buff_y = temp_buf_ptr + temp_buf_offset;

		// shift diff
		ASL_ShiftBackwardMerged_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
		<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
				count, num, channels,
				top_height, top_width,
				bottom_height, bottom_width,
				xshift, yshift,
				pad_h, pad_w,
				stride_h, stride_w,
				bottom_data, top_diff,
				shift_temp_buff_x, shift_temp_buff_y);
	}
};




template <typename DType>
struct ASL_BottomBackward<GPUDevice, DType>{
	void operator()(const GPUDevice& d,
			const int count, const int num, const int channels,
			const int top_height, const int top_width,
			const int bottom_height, const int bottom_width,
			const DType* xshift, const DType* yshift,
			const int pad_h, const int pad_w,
			const int stride_h, const int stride_w,
			const DType* bottom_data, const DType* top_diff,
			DType* bottom_backprop_ptr, DType* offset_backprop_ptr)
	{
		CudaLaunchConfig config = GetCudaLaunchConfig(count, d);

		//bottom diff
		if(stride_h==1 && stride_w==1)
		{
			ASL_BottomBackward_Stride1_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
			<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
					count, channels,
					top_height, top_width,
					bottom_height, bottom_width,
					xshift, yshift,
					pad_h, pad_w,
					top_diff, bottom_backprop_ptr);
		}
		else
		{
			ASL_BottomBackward_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
			<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
					count, channels,
					top_height, top_width,
					bottom_height, bottom_width,
					xshift, yshift,
					pad_h, pad_w,
					stride_h, stride_w,
					top_diff, bottom_backprop_ptr);
		}
	}
};



template <typename DType>
struct ApplyShiftConstraint<GPUDevice, DType>{
	void operator()(const GPUDevice& d, const int count,
			const DType* xshift_data, const DType* yshift_data,
			DType* xshift_diff, DType* yshift_diff,
			const bool normalize, const float clip_gradient)
	{
		CudaLaunchConfig config = GetCudaLaunchConfig(count, d);

	    ApplyShiftConstraint_GPUKernel<DType>  // NOLINT_NEXT_LINE(whitespace/operators)
	    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
	    		count, xshift_data, yshift_data, xshift_diff, yshift_diff, normalize, clip_gradient);

	}
};



// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

template <typename DType>
struct setZero<GPUDevice, DType>{
    void operator() (const GPUDevice& d, const int n, DType* out){
    	CUDA_CHECK(cudaMemset(out, 0, sizeof(DType) * n));
    }
};


template <typename DType>
struct setValue<GPUDevice, DType>{
    void operator() (const GPUDevice& d, const int n, DType val, DType* out){
        CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
        setValueKernel<DType> <<< config.block_count, config.thread_per_block, 0, d.stream() >>>(n, val, out);
    }
};


}  // namespace functor

#define DECLARE_GPU_SPEC(DType)                                  \
    template struct functor::ASLForward<GPUDevice, DType>; \
    template struct functor::ASL_ShiftBackward<GPUDevice, DType>; \
    template struct functor::ASL_BottomBackward<GPUDevice, DType>; \
    template struct functor::ApplyShiftConstraint<GPUDevice, DType>; \
    template struct functor::setZero<GPUDevice, DType>; \
    template struct functor::setValue<GPUDevice, DType>;
    
// extern template struct Copy<GPUDevice, T>;
TF_CALL_float(DECLARE_GPU_SPEC);
TF_CALL_double(DECLARE_GPU_SPEC);

// TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC

}  // namespace tensorflow

// #endif  // GOOGLE_CUDA
