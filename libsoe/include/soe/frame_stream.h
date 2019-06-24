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

    void input_frame(Frame frame);

    bool has_output() const;
    Frame output_frame();

private:
    double target_fps_;

    Frame frame_a_;
    Frame frame_b_;
    int frames_count_ = 0;
};

}  // namespace soe
