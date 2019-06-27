#pragma once

#include <opencv2/core/mat.hpp>

namespace soe {

struct Frame {
    cv::Mat picture;
    double timestamp;
};

}  // namespace soe
