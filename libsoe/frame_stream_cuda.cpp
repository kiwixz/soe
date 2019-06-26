#include "soe/frame_stream_cuda.h"
#include "soe/flow_to_map.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

namespace soe {

FrameStreamCuda::FrameStreamCuda(double target_fps, FarnebackSettings settings) :
    target_fps_{target_fps}
{
    frame_a_.timestamp = -1;
    frame_b_.timestamp = -1;

    farneback_ = cv::cuda::FarnebackOpticalFlow::create(settings.num_levels,
                                                        settings.pyr_scale,
                                                        settings.fast_pyramids,
                                                        settings.win_size,
                                                        settings.num_iters,
                                                        settings.poly_n,
                                                        settings.poly_sigma,
                                                        settings.flags);
}

bool FrameStreamCuda::has_output() const
{
    if (frame_a_.timestamp < 0)  // we dont have 2 frames yet
        return false;

    double next_frame_ts = frames_count_ / target_fps_;
    return next_frame_ts < frame_b_.timestamp;
}

void FrameStreamCuda::input_frame(Frame frame)
{
    frame_a_ = std::move(frame_b_);
    frame_b_ = std::move(frame);
}

FrameStreamCuda::Frame FrameStreamCuda::output_frame()
{
    Frame frame;
    frame.timestamp = frames_count_ / target_fps_;

    double t = (frame.timestamp - frame_a_.timestamp) / (frame_b_.timestamp - frame_a_.timestamp);  // how close of frame_b_ we are [0;1]

    cv::cuda::GpuMat from;
    cv::cuda::cvtColor(frame_a_.picture, from, cv::COLOR_BGR2GRAY);
    cv::cuda::GpuMat to;
    cv::cuda::cvtColor(frame_b_.picture, to, cv::COLOR_BGR2GRAY);

    if (last_flow_.size() != from.size())
        last_flow_ = {from.size(), CV_32FC2};

    // calculate backward dense optical flow
    farneback_->calc(to, from, last_flow_);

    cv::cuda::GpuMat x_map{from.size(), CV_32FC1};
    cv::cuda::GpuMat y_map{from.size(), CV_32FC1};
    cuda::flow_to_map(last_flow_, x_map, y_map, t);

    cv::cuda::remap(frame_a_.picture, frame.picture, x_map, y_map, cv::INTER_NEAREST, cv::BORDER_REPLICATE);

    ++frames_count_;
    return frame;
}

}  // namespace soe
