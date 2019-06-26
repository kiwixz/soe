#pragma once

#include <opencv2/core/cuda.hpp>

namespace soe {
namespace cuda {

void flow_to_map(const cv::cuda::GpuMat& flow,
                 cv::cuda::GpuMat& x_map, cv::cuda::GpuMat& y_map,
                 double t);

}
}  // namespace soe
