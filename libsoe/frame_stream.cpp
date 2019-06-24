#include "soe/frame_stream.h"

namespace soe {

FrameStream::FrameStream(double target_fps) :
    target_fps_{target_fps}
{
    frame_a_.timestamp = -1;
    frame_b_.timestamp = -1;
}

void FrameStream::input_frame(Frame frame)
{
    frame_a_ = std::move(frame_b_);
    frame_b_ = std::move(frame);
}

bool FrameStream::has_output() const
{
    if (frame_a_.timestamp < 0)  // we dont have 2 frames yet
        return false;

    double next_frame_ts = frames_count_ / target_fps_;
    return next_frame_ts < frame_b_.timestamp;
}

FrameStream::Frame FrameStream::output_frame()
{
    ++frames_count_;
    return frame_a_;
}

}  // namespace soe
