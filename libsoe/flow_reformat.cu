#include "soe/flow_kernels.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace soe {
namespace cuda {
namespace {

__global__ void kernel(const cv::cuda::PtrStep<float2> flow,
                       const cv::cuda::PtrStepb flow_status,
                       int2 scale,
                       cv::cuda::PtrStepSz<float2> flow_rel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= flow_rel.cols || y >= flow_rel.rows)
        return;

    int index = y * flow_rel.cols + x;
    float2 flow_xy = flow(0, index);
    float2 rel;
    if (flow_status(0, index) == 1) {
        rel.x = flow_xy.x - x * scale.x;
        rel.y = flow_xy.y - y * scale.y;
    }
    else {
        rel.x = 0;
        rel.y = 0;
    }
    flow_rel(y, x) = rel;
}

}  // namespace


void flow_reformat(const cv::cuda::GpuMat& flow,
                   const cv::cuda::GpuMat& flow_status,
                   cv::Size picture_size,
                   cv::cuda::GpuMat& flow_rel,
                   cv::cuda::Stream cuda_stream)
{
    dim3 threads{64, 16};
    dim3 blocks{static_cast<unsigned>(std::ceil(flow_rel.size().width / static_cast<double>(threads.x))),
                static_cast<unsigned>(std::ceil(flow_rel.size().height / static_cast<double>(threads.y)))};
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cuda_stream);
    kernel<<<blocks, threads, 0, stream>>>(flow, flow_status,
                                           {picture_size.width / flow_rel.size().width,
                                            picture_size.height / flow_rel.size().height},
                                           flow_rel);

#if DEBUG
    cuda_stream.waitForCompletion();  // will abort if there is any error
#endif
}

}  // namespace cuda
}  // namespace soe
