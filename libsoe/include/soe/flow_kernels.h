#pragma once

#include <opencv2/core/cuda.hpp>

namespace soe {
namespace cuda {

void flow_reformat(const cv::cuda::GpuMat& flow,
                   const cv::cuda::GpuMat& flow_status,
                   cv::Size picture_size,
                   cv::cuda::GpuMat& flow_rel,
                   cv::cuda::Stream cuda_stream);

void flow_to_map(const cv::cuda::GpuMat& flow,
                 double t,
                 cv::cuda::GpuMat& x_map, cv::cuda::GpuMat& y_map,
                 cv::cuda::Stream cuda_stream);

}  // namespace cuda
}  // namespace soe
