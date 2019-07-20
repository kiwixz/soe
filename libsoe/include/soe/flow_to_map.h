#pragma once

#include <opencv2/core/cuda.hpp>

namespace soe {
namespace cuda {

void flow_to_map(const cv::cuda::GpuMat& flow, const cv::cuda::GpuMat& flow_status,
                 cv::Size picture_size, double t,
                 cv::cuda::GpuMat& x_map, cv::cuda::GpuMat& y_map,
                 cv::cuda::Stream cuda_stream = cv::cuda::Stream::Null());

}
}  // namespace soe
