#pragma once

#include <opencv2/core/mat.hpp>

namespace soe {

struct FrameStream {
    struct Frame {
        cv::Mat picture;
        double timestamp;
    };

    FrameStream() = default;
    explicit FrameStream(double target_fps);

    [[nodiscard]] bool has_output() const;

    void input_frame(Frame frame);
    Frame output_frame();

private:
    double target_fps_;

    Frame frame_a_;
    Frame frame_b_;
    cv::Mat last_flow_;
    int frames_count_ = 0;
};

}  // namespace soe
