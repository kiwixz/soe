#include "soe/picture.h"

namespace soe {

PictureYV12::PictureYV12(int width, int height) :
    width_{width}, height_{height}
{
    int nr_pixels = width_ * height_;
    data_.resize(nr_pixels);
    v_ = data_.data() + nr_pixels;
    u_ = data_.data() + nr_pixels + nr_pixels / 4;
}

int PictureYV12::width() const
{
    return width_;
}

int PictureYV12::height() const
{
    return height_;
}

uint8_t* PictureYV12::y()
{
    return data_.data();
}
const uint8_t* PictureYV12::y() const
{
    return data_.data();
}

uint8_t* PictureYV12::u()
{
    return u_;
}
const uint8_t* PictureYV12::u() const
{
    return u_;
}

uint8_t* PictureYV12::v()
{
    return v_;
}
const uint8_t* PictureYV12::v() const
{
    return v_;
}

}  // namespace soe
