#pragma once

#include "soe/farneback_settings.h"
#include "soe/frame.h"
#include "utils/vec.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>

namespace soe {

struct FrameStreamCuda {
    struct Settings {
        double target_fps = 60.0;
        double poly_sigma = .5;
    };

    FrameStreamCuda() = default;
    FrameStreamCuda(cv::Size picture_size, double target_fps, utils::Vec2d scale, const FarnebackSettings& settings);

    [[nodiscard]] bool has_output() const;

    void input_frame(Frame frame);
    Frame output_frame();

private:
    struct GpuFrame {
        GpuFrame() = default;

        cv::cuda::GpuMat picture;
        double timestamp;

        cv::cuda::GpuMat gray;
        cv::cuda::GpuMat scaled;
    };

    struct Flow {
        Flow(cv::Size picture_size, utils::Vec2d scale, const FarnebackSettings& settings);

        cv::Ptr<cv::cuda::FarnebackOpticalFlow> farneback;
        cv::cuda::GpuMat last_flow;

        // just to avoid realloc
        cv::cuda::GpuMat last_flow_scaled;
        cv::cuda::GpuMat remap;
    };


    double target_fps_;

    cv::cuda::Stream cuda_stream_;
    GpuFrame frame_a_;
    GpuFrame frame_b_;
    Flow flow_ab_;
    Flow flow_ba_;
    bool is_flow_fresh_ = false;
    int frames_count_ = 0;

    // just to avoid realloc
    cv::cuda::GpuMat x_map_;
    cv::cuda::GpuMat y_map_;
    cv::cuda::GpuMat frame_gpu_;
};

}  // namespace soe
