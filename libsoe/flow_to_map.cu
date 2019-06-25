#include <opencv2/core/cuda.hpp>

namespace soe {
namespace cuda {
namespace {

__global__ void flow_to_map_kernel(const cv::cuda::PtrStepSzf flow, cv::cuda::PtrStepSzf map, float t, int pixels_per_thread)
{
    int base_x = threadIdx.x * pixels_per_thread;
    int y = blockIdx.x;

    int next_thread_x = base_x + pixels_per_thread;
    if (next_thread_x - 1 > flow.cols)
        return;

    const float* flow_row = flow.ptr(y);
    float* map_row = map.ptr(y);

    for (int x = base_x; x < next_thread_x; ++x) {
        map_row[x * 2 + 0] = x + flow_row[x * 2 + 0] * t;
        map_row[x * 2 + 1] = y + flow_row[x * 2 + 1] * t;
    }
}

}  // namespace


cv::cuda::GpuMat flow_to_map(const cv::cuda::GpuMat& flow, double t)
{
    int nr_threads = flow.cols;
    while (nr_threads > 1024)
        nr_threads /= 2;

    cv::cuda::GpuMat map{flow.size(), CV_32FC2};
    int pixels_per_thread = flow.cols / nr_threads;
    flow_to_map_kernel<<<flow.rows, nr_threads>>>(flow, map, t, pixels_per_thread);

    return map;
}

}  // namespace cuda
}  // namespace soe
