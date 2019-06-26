#include "soe/flow_to_map.h"

namespace soe {
namespace cuda {
namespace {

__global__ void flow_to_map_kernel(const cv::cuda::PtrStepSzf flow,
                                   cv::cuda::PtrStepSzf x_map, cv::cuda::PtrStepSzf y_map,
                                   float t, int pixels_per_thread)
{
    int base_x = threadIdx.x * pixels_per_thread;
    int y = blockIdx.x;

    int next_thread_x = base_x + pixels_per_thread;
    if (next_thread_x - 1 > flow.cols)
        return;

    const float* flow_row = flow.ptr(y);
    float* x_map_row = x_map.ptr(y);
    float* y_map_row = y_map.ptr(y);

    for (int x = base_x; x < next_thread_x; ++x) {
        x_map_row[x] = x + flow_row[x * 2 + 0] * t;
        y_map_row[x] = y + flow_row[x * 2 + 1] * t;
    }
}

}  // namespace


void flow_to_map(const cv::cuda::GpuMat& flow,
                 cv::cuda::GpuMat& x_map, cv::cuda::GpuMat& y_map,
                 double t)
{
    int nr_threads = flow.cols;
    while (nr_threads > 1024)
        nr_threads /= 2;

    int pixels_per_thread = flow.cols / nr_threads;
    flow_to_map_kernel<<<flow.rows, nr_threads>>>(flow, x_map, y_map, t, pixels_per_thread);
}

}  // namespace cuda
}  // namespace soe
