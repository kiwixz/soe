#include "soe/flow_to_map.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace soe {
namespace cuda {
namespace {

__global__ void flow_to_map_kernel(const cv::cuda::PtrStep<float2> flow, const cv::cuda::PtrStepb flow_status,
                                   int2 picture_size, float t,
                                   cv::cuda::PtrStepf x_map, cv::cuda::PtrStepf y_map)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= picture_size.x || y >= picture_size.y)
        return;

    bool found = flow_status(0, y * picture_size.x + x);
    float2 flow_xy = flow(0, y * picture_size.x + x);
    x_map(y, x) = found * (x + t * (flow_xy.x - x)) + !found * x;
    y_map(y, x) = found * (y + t * (flow_xy.y - y)) + !found * y;
}

}  // namespace


void flow_to_map(const cv::cuda::GpuMat& flow, const cv::cuda::GpuMat& flow_status,
                 cv::Size picture_size, double t,
                 cv::cuda::GpuMat& x_map, cv::cuda::GpuMat& y_map,
                 cv::cuda::Stream cuda_stream)
{
    dim3 threads{64, 16};
    dim3 blocks{static_cast<unsigned>(std::ceil(picture_size.width / static_cast<double>(threads.x))),
                static_cast<unsigned>(std::ceil(picture_size.height / static_cast<double>(threads.y)))};
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cuda_stream);
    flow_to_map_kernel<<<blocks, threads, 0, stream>>>(flow, flow_status,
                                                       {picture_size.width, picture_size.height}, static_cast<float>(t),
                                                       x_map, y_map);

#if DEBUG
    cuda_stream.waitForCompletion();  // will abort if there is any error
#endif
}

}  // namespace cuda
}  // namespace soe
