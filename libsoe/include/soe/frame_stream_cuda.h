#pragma once

#include "soe/farneback_settings.h"
#include "soe/frame.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>

namespace soe {

struct FrameStreamCuda {
    struct GpuFrame {
        GpuFrame() = default;

        cv::cuda::GpuMat picture;
        double timestamp;

        cv::cuda::GpuMat gray;
    };

    struct Settings {
        double target_fps = 60.0;
        double poly_sigma = .5;
    };

    FrameStreamCuda() = default;
    FrameStreamCuda(cv::Size picture_size, double target_fps, FarnebackSettings settings);

    [[nodiscard]] bool has_output() const;

    void input_frame(Frame frame);
    Frame output_frame();

private:
    cv::Size picture_size_;
    double target_fps_;

    cv::cuda::Stream cuda_stream_;
    GpuFrame frame_a_;
    GpuFrame frame_b_;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optflow_;
    cv::cuda::GpuMat grid_;
    int frames_count_ = 0;

    cv::cuda::GpuMat flow_;
    cv::cuda::GpuMat flow_status_;
    bool is_flow_fresh_ = false;

    // just to avoid realloc
    cv::cuda::GpuMat x_map_;
    cv::cuda::GpuMat y_map_;
    cv::cuda::GpuMat frame_gpu_;
};

}  // namespace soe
