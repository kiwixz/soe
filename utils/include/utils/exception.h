#pragma once

#include <fmt/format.h>
#include <stdexcept>
#include <string_view>

namespace utils {

struct Exception : std::runtime_error {
    explicit Exception(std::string_view what);

    template <typename... Args>
    explicit Exception(std::string_view format, Args... args);
};


template <typename... Args>
Exception::Exception(std::string_view format, Args... args) :
    Exception{fmt::format(format, std::forward<Args>(args)...)}
{}

}  // namespace utils
