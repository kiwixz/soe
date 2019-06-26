#pragma once

#include "soe/farneback_settings.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>

namespace soe {

struct FrameStreamCuda {
    struct Frame {
        cv::cuda::GpuMat picture;
        double timestamp;
    };

    struct Settings {
        double target_fps = 60.0;
        double poly_sigma = .5;
    };

    FrameStreamCuda() = default;
    FrameStreamCuda(double target_fps, FarnebackSettings settings);

    [[nodiscard]] bool has_output() const;

    void input_frame(Frame frame);
    Frame output_frame();

private:
    double target_fps_;

    Frame frame_a_;
    Frame frame_b_;
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farneback_;
    cv::cuda::GpuMat last_flow_;
    int frames_count_ = 0;
};

}  // namespace soe
