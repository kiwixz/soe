#include "utils/rand.h"
#include <algorithm>
#include <random>

namespace utils {
namespace {

uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

}  // namespace


Rand64::Rand64()
{
    std::random_device source;
    std::generate(state_.begin(), state_.end(), [&] {
        return static_cast<uint64_t>(source()) << 32 | source();
    });
}

uint64_t Rand64::operator()()
{
    return gen();
}

uint64_t Rand64::gen()
{
    // xoshiro256starstar
    uint64_t* s = state_.data();

    const uint64_t result_starstar = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result_starstar;
}


RandF::RandF()
{
    std::random_device source;
    std::generate(state_.begin(), state_.end(), [&] {
        return static_cast<uint64_t>(source()) << 32 | source();
    });
}

double RandF::operator()()
{
    return gen();
}

double RandF::gen()
{
    // xoshiro256plus
    uint64_t* s = state_.data();

    const uint64_t result_plus = s[0] + s[3];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    uint64_t rnd = result_plus >> 3;     // last 3 bits are not really random
    return rnd / 2305843009213694001.0;  // divide by 2^61+1 to map to [0,1)
}

}  // namespace utils
