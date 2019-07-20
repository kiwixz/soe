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

FrameStreamCuda::FrameStreamCuda(cv::Size picture_size, double target_fps, utils::Vec2d scale, FarnebackSettings settings) :
    target_fps_{target_fps}
{
    frame_a_.timestamp = -1;
    frame_b_.timestamp = -1;

    last_flow_scaled_ = {cv::Size{static_cast<int>(picture_size.width * scale.x),
                                  static_cast<int>(picture_size.height * scale.y)},
                         CV_32FC2};
    last_flow_ = {picture_size, CV_32FC2};
    x_map_ = {picture_size, CV_32FC1};
    y_map_ = {picture_size, CV_32FC1};

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
    is_flow_fresh_ = false;
    frame_a_ = std::move(frame_b_);
    frame_b_ = {};  // GpuMat is like a shared_ptr without move, so we must create another one
    frame_b_.picture.upload(frame.picture, cuda_stream_);
    frame_b_.timestamp = frame.timestamp;
    cv::cuda::cvtColor(frame_b_.picture, frame_b_.gray, cv::COLOR_BGR2GRAY, 0, cuda_stream_);
    cv::cuda::resize(frame_b_.gray, frame_b_.scaled, last_flow_scaled_.size(), 0.0, 0.0, cv::INTER_LINEAR, cuda_stream_);
}

Frame FrameStreamCuda::output_frame()
{
    Frame frame;
    frame.timestamp = frames_count_ / target_fps_;

    double t = (frame.timestamp - frame_a_.timestamp) / (frame_b_.timestamp - frame_a_.timestamp);  // how close of frame_b_ we are [0;1]

    if (!is_flow_fresh_) {
        // calculate backward dense optical flow
        cuda_stream_.waitForCompletion();  // seems necessary to avoid artifacts
        farneback_->calc(frame_b_.scaled, frame_a_.scaled, last_flow_scaled_, cuda_stream_);
        resize_2c(last_flow_scaled_, last_flow_, last_flow_.size(), 0.0, 0.0, cv::INTER_LINEAR, cuda_stream_);
        is_flow_fresh_ = true;
    }

    cuda::flow_to_map(last_flow_, t, x_map_, y_map_, cuda_stream_);
    cv::cuda::remap(frame_a_.picture, frame_gpu_, x_map_, y_map_, cv::INTER_NEAREST, cv::BORDER_REPLICATE, {}, cuda_stream_);
    frame_gpu_.download(frame.picture, cuda_stream_);

    ++frames_count_;
    cuda_stream_.waitForCompletion();
    return frame;
}

}  // namespace soe
