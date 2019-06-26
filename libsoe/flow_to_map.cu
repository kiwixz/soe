#include "soe/flow_to_map.h"

namespace soe {
namespace cuda {
namespace {

__global__ void flow_to_map_kernel(const cv::cuda::PtrStepSz<float2> flow,
                                   cv::cuda::PtrStepSzf x_map, cv::cuda::PtrStepSzf y_map,
                                   float t)
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


void flow_to_map(const cv::cuda::GpuMat& flow,
                 cv::cuda::GpuMat& x_map, cv::cuda::GpuMat& y_map,
                 double t)
{
    dim3 threads{64, 16};
    dim3 blocks{static_cast<unsigned>(std::ceil(flow.size().width / static_cast<double>(threads.x))),
                static_cast<unsigned>(std::ceil(flow.size().height / static_cast<double>(threads.y)))};
    flow_to_map_kernel<<<blocks, threads>>>(flow, x_map, y_map, static_cast<float>(t));
}

}  // namespace cuda
}  // namespace soe
