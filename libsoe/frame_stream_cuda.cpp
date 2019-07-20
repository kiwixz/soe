#include "soe/frame_stream_cuda.h"
#include "soe/flow_to_map.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

namespace soe {

FrameStreamCuda::FrameStreamCuda(cv::Size picture_size, double target_fps, FarnebackSettings settings) :
    picture_size_{picture_size},
    target_fps_{target_fps}
{
    frame_a_.timestamp = -1;
    frame_b_.timestamp = -1;

    constexpr int speed_x = 1;
    constexpr int speed_y = 1;
    int speed_width = picture_size_.width / speed_x;
    int speed_height = picture_size_.height / speed_y;
    cv::Mat grid{cv::Size{speed_width * speed_height, 1}, CV_32FC2};
    for (int y = 0; y < speed_height; ++y)
        for (int x = 0; x < speed_width; ++x)
            grid.at<cv::Point2f>(y * speed_width + x) = {static_cast<float>(x * speed_x),
                                                         static_cast<float>(y * speed_y)};
    grid_.upload(grid);

    flow_ = {grid_.size(), CV_32FC2};
    flow_status_ = {flow_.size(), CV_8UC1};
    x_map_ = {picture_size, CV_32FC1};
    y_map_ = {picture_size, CV_32FC1};

    optflow_ = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(13, 13), 3, 30, true);
#if 0
    optflow_ = cv::cuda::FarnebackOpticalFlow::create(settings.num_levels,
                                                        settings.pyr_scale,
                                                        settings.fast_pyramids,
                                                        settings.win_size,
                                                        settings.num_iters,
                                                        settings.poly_n,
                                                        settings.poly_sigma,
                                                        settings.flags);
#endif
}

bool FrameStreamCuda::has_output() const
{
    if (frame_a_.timestamp < 0)  // we dont have 2 frames yet
        return false;

    double next_frame_ts = frames_count_ / target_fps_;
    return next_frame_ts < frame_b_.timestamp;
}

void FrameStreamCuda::input_frame(Frame frame)
{
    assert(frame.picture.size() == picture_size_);
    is_flow_fresh_ = false;
    frame_a_ = std::move(frame_b_);
    frame_b_ = {};  // GpuMat is like a shared_ptr without move, so we must create another one
    frame_b_.picture.upload(frame.picture, cuda_stream_);
    frame_b_.timestamp = frame.timestamp;
    //cv::cuda::cvtColor(frame_b_.picture, frame_b_.gray, cv::COLOR_BGR2GRAY, 0, cuda_stream_);
    frame_b_.gray = frame_b_.picture;
}

Frame FrameStreamCuda::output_frame()
{
    Frame frame;
    frame.timestamp = frames_count_ / target_fps_;

    double t = (frame.timestamp - frame_a_.timestamp) / (frame_b_.timestamp - frame_a_.timestamp);  // how close of frame_b_ we are [0;1]
    cv::Size picture_size = frame_a_.picture.size();

    if (!is_flow_fresh_) {
        // calculate backward optical flow
        optflow_->calc(frame_b_.gray, frame_a_.gray,
                       grid_, flow_, flow_status_, cv::noArray(), cuda_stream_);
        is_flow_fresh_ = true;
    }

    cuda::flow_to_map(flow_, flow_status_, picture_size, t, x_map_, y_map_, cuda_stream_);
    cv::cuda::remap(frame_a_.picture, frame_gpu_, x_map_, y_map_, cv::INTER_NEAREST, cv::BORDER_REPLICATE, {}, cuda_stream_);
    frame_gpu_.download(frame.picture, cuda_stream_);

    ++frames_count_;
    cuda_stream_.waitForCompletion();
    return frame;
}

}  // namespace soe
