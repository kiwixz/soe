#pragma once

#include <vector>

namespace soe {

struct PictureYV12 {
    PictureYV12() = default;
    PictureYV12(int width, int height);

    int width() const;
    int height() const;

    uint8_t* y();
    uint8_t const* y() const;
    uint8_t* u();
    uint8_t const* u() const;
    uint8_t* v();
    uint8_t const* v() const;

private:
    int width_;
    int height_;
    std::vector<uint8_t> data_;
    uint8_t* v_;
    uint8_t* u_;
};

}  // namespace soe
