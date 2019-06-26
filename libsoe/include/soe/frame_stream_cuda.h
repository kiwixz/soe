#pragma once

#include <opencv2/core/cuda.hpp>

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
    explicit FrameStreamCuda(Settings settings);

    [[nodiscard]] bool has_output() const;

    void input_frame(Frame frame);
    Frame output_frame();

private:
    Settings settings_;

    Frame frame_a_;
    Frame frame_b_;
    cv::cuda::GpuMat last_flow_;
    int frames_count_ = 0;
};

}  // namespace soe
