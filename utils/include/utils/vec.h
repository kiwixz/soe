#pragma once

namespace utils {

template <typename TElement>
struct Vec2 {
    using Element = TElement;

    Element x;
    Element y;
};


using Vec2i = Vec2<int>;
using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;

}  // namespace utils
