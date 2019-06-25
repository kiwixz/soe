#pragma once

#include <opencv2/core/mat.hpp>

namespace soe {

struct FrameStream {
    struct Frame {
        cv::Mat picture;
        double timestamp;
    };

    struct Settings {
        double target_fps = 60.0;
        double poly_sigma = .5;
    };

    FrameStream() = default;
    explicit FrameStream(Settings settings);

    [[nodiscard]] bool has_output() const;

    void input_frame(Frame frame);
    Frame output_frame();

private:
    Settings settings_;

    Frame frame_a_;
    Frame frame_b_;
    cv::Mat last_flow_;
    int frames_count_ = 0;
};

}  // namespace soe
