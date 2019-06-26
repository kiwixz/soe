#include "soe/frame_stream.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace soe {

FrameStream::FrameStream(double target_fps, FarnebackSettings settings) :
    target_fps_{target_fps}
{
    frame_a_.timestamp = -1;
    frame_b_.timestamp = -1;

    farneback_ = cv::FarnebackOpticalFlow::create(settings.num_levels,
                                                  settings.pyr_scale,
                                                  settings.fast_pyramids,
                                                  settings.win_size,
                                                  settings.num_iters,
                                                  settings.poly_n,
                                                  settings.poly_sigma,
                                                  settings.flags);
}

bool FrameStream::has_output() const
{
    if (frame_a_.timestamp < 0)  // we dont have 2 frames yet
        return false;

    double next_frame_ts = frames_count_ / target_fps_;
    return next_frame_ts < frame_b_.timestamp;
}

void FrameStream::input_frame(Frame frame)
{
    frame_a_ = std::move(frame_b_);
    frame_b_ = std::move(frame);
}

FrameStream::Frame FrameStream::output_frame()
{
    Frame frame;
    frame.timestamp = frames_count_ / target_fps_;

    double t = (frame.timestamp - frame_a_.timestamp) / (frame_b_.timestamp - frame_a_.timestamp);  // how close of frame_b_ we are [0;1]

    cv::Mat from;
    cv::cvtColor(frame_a_.picture, from, cv::COLOR_BGR2GRAY);
    cv::Mat to;
    cv::cvtColor(frame_b_.picture, to, cv::COLOR_BGR2GRAY);

    if (last_flow_.size() != from.size())
        last_flow_ = {from.size(), CV_32FC2};

    // calculate backward dense optical flow
    farneback_->calc(to, from, last_flow_);

    cv::Mat map{last_flow_.size(), CV_32FC2};
    for (int y = 0; y < map.rows; ++y)
        for (int x = 0; x < map.cols; ++x) {
            const auto& f = last_flow_.at<cv::Point2f>(y, x);
            map.at<cv::Point2f>(y, x) = {static_cast<float>(x + f.x * t),
                                         static_cast<float>(y + f.y * t)};
        }

    cv::remap(frame_a_.picture, frame.picture, map, {}, cv::INTER_NEAREST, cv::BORDER_REPLICATE);

    ++frames_count_;
    return frame;
}

}  // namespace soe
