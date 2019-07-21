#include "soe/frame_stream_cuda.h"
#include "soe/flow_to_map.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

namespace soe {
namespace {

// opencv cuda did not implement this one
void resize_2c(const cv::cuda::GpuMat& a, cv::cuda::GpuMat& b,
               cv::Size size, double fx, double fy,
               int interpolation, cv::cuda::Stream& stream)
{
    std::array<cv::cuda::GpuMat, 2> a_split;
    std::array<cv::cuda::GpuMat, 2> b_split;
    cv::cuda::split(a, a_split.data(), stream);
    cv::cuda::resize(a_split[0], b_split[0], size, fx, fy, interpolation, stream);
    cv::cuda::resize(a_split[1], b_split[1], size, fx, fy, interpolation, stream);
    cv::cuda::merge(b_split.data(), b_split.size(), b, stream);
}

}  // namespace


FrameStreamCuda::FrameStreamCuda(cv::Size picture_size, double target_fps, utils::Vec2d scale, const FarnebackSettings& settings) :
    target_fps_{target_fps},
    flow_ab_{picture_size, scale, settings},
    flow_ba_{picture_size, scale, settings}
{
    frame_a_.timestamp = -1;
    frame_b_.timestamp = -1;

    x_map_ = {picture_size, CV_32FC1};
    y_map_ = {picture_size, CV_32FC1};
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
    is_flow_fresh_ = false;
    frame_a_ = std::move(frame_b_);
    frame_b_ = {};  // GpuMat is like a shared_ptr without move, so we must create another one
    frame_b_.picture.upload(frame.picture, cuda_stream_);
    frame_b_.timestamp = frame.timestamp;
    cv::cuda::cvtColor(frame_b_.picture, frame_b_.gray, cv::COLOR_BGR2GRAY, 0, cuda_stream_);
    cv::cuda::resize(frame_b_.gray, frame_b_.scaled, flow_ab_.last_flow_scaled.size(), 0.0, 0.0, cv::INTER_LINEAR, cuda_stream_);
}

Frame FrameStreamCuda::output_frame()
{
    Frame frame;
    frame.timestamp = frames_count_ / target_fps_;

    double t = (frame.timestamp - frame_a_.timestamp) / (frame_b_.timestamp - frame_a_.timestamp);  // how close of frame_b_ we are [0;1]

    if (!is_flow_fresh_) {
        // calculate forward and backward dense optical flow
        cuda_stream_.waitForCompletion();  // seems necessary to avoid artifacts
        flow_ab_.farneback->calc(frame_a_.scaled, frame_b_.scaled, flow_ab_.last_flow_scaled, cuda_stream_);
        resize_2c(flow_ab_.last_flow_scaled, flow_ab_.last_flow, flow_ab_.last_flow.size(), 0.0, 0.0, cv::INTER_LINEAR, cuda_stream_);
        flow_ba_.farneback->calc(frame_b_.scaled, frame_a_.scaled, flow_ba_.last_flow_scaled, cuda_stream_);
        resize_2c(flow_ba_.last_flow_scaled, flow_ba_.last_flow, flow_ba_.last_flow.size(), 0.0, 0.0, cv::INTER_LINEAR, cuda_stream_);
        is_flow_fresh_ = true;
    }

    cuda::flow_to_map(flow_ab_.last_flow, 1.0 - t, x_map_, y_map_, cuda_stream_);
    cv::cuda::remap(frame_b_.picture, flow_ab_.remap, x_map_, y_map_, cv::INTER_LINEAR, cv::BORDER_REPLICATE, {}, cuda_stream_);
    cuda::flow_to_map(flow_ba_.last_flow, t, x_map_, y_map_, cuda_stream_);
    cv::cuda::remap(frame_a_.picture, flow_ba_.remap, x_map_, y_map_, cv::INTER_LINEAR, cv::BORDER_REPLICATE, {}, cuda_stream_);
    cv::cuda::addWeighted(flow_ab_.remap, t, flow_ba_.remap, 1.0 - t, 0.0, frame_gpu_, -1, cuda_stream_);
    frame_gpu_.download(frame.picture, cuda_stream_);

    ++frames_count_;
    cuda_stream_.waitForCompletion();
    return frame;
}


FrameStreamCuda::Flow::Flow(cv::Size picture_size, utils::Vec2d scale, const FarnebackSettings& settings)
{
    last_flow_scaled = {cv::Size{static_cast<int>(picture_size.width * scale.x),
                                 static_cast<int>(picture_size.height * scale.y)},
                        CV_32FC2};
    last_flow = {picture_size, CV_32FC2};

    farneback = cv::cuda::FarnebackOpticalFlow::create(settings.num_levels,
                                                       settings.pyr_scale,
                                                       settings.fast_pyramids,
                                                       settings.win_size,
                                                       settings.num_iters,
                                                       settings.poly_n,
                                                       settings.poly_sigma,
                                                       settings.flags);
}

}  // namespace soe
