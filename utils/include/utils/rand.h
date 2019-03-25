#pragma once

#include <array>
#include <cstdint>

namespace utils {

struct Rand64 {
    Rand64();

    uint64_t operator()();

    uint64_t gen();

private:
    std::array<uint64_t, 4> state_;
};

struct RandF {
    RandF();

    double operator()();

    double gen();

private:
    std::array<uint64_t, 4> state_;
};

}  // namespace utils
