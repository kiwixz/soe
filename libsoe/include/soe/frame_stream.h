#pragma once

#include "soe/farneback_settings.h"
#include "soe/frame.h"
#include <opencv2/video/tracking.hpp>

namespace soe {

struct FrameStream {
    struct Settings {
        double target_fps = 60.0;
        double poly_sigma = .5;
    };

    FrameStream() = default;
    FrameStream(double target_fps, FarnebackSettings settings);

    [[nodiscard]] bool has_output() const;

    void input_frame(Frame frame);
    Frame output_frame();

private:
    double target_fps_;

    Frame frame_a_;
    Frame frame_b_;
    cv::Ptr<cv::FarnebackOpticalFlow> optflow_;
    cv::Mat flow_;
    int frames_count_ = 0;
};

}  // namespace soe
