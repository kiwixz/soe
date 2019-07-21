#pragma once

#include <opencv2/video/tracking.hpp>

namespace soe {

struct FarnebackSettings {
    int num_levels = 5;
    double pyr_scale = 0.5;
    bool fast_pyramids = false;
    int win_size = 26;
    int num_iters = 10;
    int poly_n = 5;
    double poly_sigma = 1.1;
    int flags = cv::OPTFLOW_USE_INITIAL_FLOW;
};

}  // namespace soe
