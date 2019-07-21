#include "soe/flow_to_map.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace soe {
namespace cuda {
namespace {

__global__ void flow_to_map_kernel(const cv::cuda::PtrStepSz<float2> flow, float t,
                                   cv::cuda::PtrStepf x_map, cv::cuda::PtrStepf y_map)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= flow.cols || y >= flow.rows)
        return;

    float2 flow_xy = flow(y, x);
    x_map(y, x) = x + flow_xy.x * t;
    y_map(y, x) = y + flow_xy.y * t;
}

}  // namespace


void flow_to_map(const cv::cuda::GpuMat& flow, double t,
                 cv::cuda::GpuMat& x_map, cv::cuda::GpuMat& y_map,
                 cv::cuda::Stream& cuda_stream)
{
    dim3 threads{64, 16};
    dim3 blocks{static_cast<unsigned>(std::ceil(flow.size().width / static_cast<double>(threads.x))),
                static_cast<unsigned>(std::ceil(flow.size().height / static_cast<double>(threads.y)))};
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cuda_stream);
    flow_to_map_kernel<<<blocks, threads, 0, stream>>>(flow, static_cast<float>(t), x_map, y_map);
}

}  // namespace cuda
}  // namespace soe
