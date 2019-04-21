#pragma once

#include <cstdint>
#include <vector>

namespace soe {

struct PictureYV12 {
    PictureYV12() = default;
    PictureYV12(int width, int height);

    [[nodiscard]] int width() const;
    [[nodiscard]] int height() const;

    [[nodiscard]] uint8_t* y();
    [[nodiscard]] const uint8_t* y() const;
    [[nodiscard]] uint8_t* u();
    [[nodiscard]] const uint8_t* u() const;
    [[nodiscard]] uint8_t* v();
    [[nodiscard]] const uint8_t* v() const;

private:
    int width_;
    int height_;
    std::vector<uint8_t> data_;
    uint8_t* v_;
    uint8_t* u_;
};

}  // namespace soe
