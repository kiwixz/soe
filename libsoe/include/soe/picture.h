#pragma once

#include <vector>

namespace soe {

struct PictureYV12 {
    PictureYV12() = default;
    PictureYV12(int width, int height);

    [[nodiscard]] int width() const;
    [[nodiscard]] int height() const;

    [[nodiscard]] uint8_t* y();
    [[nodiscard]] uint8_t const* y() const;
    [[nodiscard]] uint8_t* u();
    [[nodiscard]] uint8_t const* u() const;
    [[nodiscard]] uint8_t* v();
    [[nodiscard]] uint8_t const* v() const;

private:
    int width_;
    int height_;
    std::vector<uint8_t> data_;
    uint8_t* v_;
    uint8_t* u_;
};

}  // namespace soe
