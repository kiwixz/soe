#pragma once

#include <functional>

namespace utils {

/// Classic RAII wrapper.
/// Functions convertable to bool and evaluating to false will not be run
/// (so you can default construct with std::function and pointers safely).
template <typename TFunction = void (*)()>
struct ScopeExit {
    using Function = TFunction;

    ScopeExit() = default;
    explicit ScopeExit(Function function);
    ~ScopeExit();
    ScopeExit(ScopeExit const&) = delete;
    ScopeExit& operator=(ScopeExit const&) = delete;
    ScopeExit(ScopeExit&& other) noexcept;
    ScopeExit& operator=(ScopeExit&& other) noexcept;

private:
    Function function_{};
};


using ScopeExitGeneric = ScopeExit<std::function<void()>>;


template <typename TFunction>
ScopeExit<TFunction>::ScopeExit(Function function) :
    function_{std::move(function)}
{}

template <typename TFunction>
ScopeExit<TFunction>::~ScopeExit()
{
    if constexpr (std::is_constructible_v<bool, Function>)
        if (!function_)  // NOLINT(bugprone-suspicious-semicolon)
            return;
    function_();
}

template <typename TFunction>
ScopeExit<TFunction>::ScopeExit(ScopeExit&& other) noexcept
{
    *this = std::move(other);
}

template <typename TFunction>
ScopeExit<TFunction>& ScopeExit<TFunction>::operator=(ScopeExit<TFunction>&& other) noexcept
{
    std::swap(function_, other.function_);
    return *this;
}

}  // namespace utils
