#pragma once

#include <memory>

namespace utils {

template <void (*TInit)(), void (*TCleanup)()>
struct SharedHandle {
    static constexpr void (*init)() = TInit;
    static constexpr void (*cleanup)() = TCleanup;

    SharedHandle();

private:
    std::shared_ptr<void> instance_;

    static std::shared_ptr<void> get_instance();
};


template <void (*TInit)(), void (*TCleanup)()>
SharedHandle<TInit, TCleanup>::SharedHandle() :
    instance_{get_instance()}
{}

template <void (*TInit)(), void (*TCleanup)()>
std::shared_ptr<void> SharedHandle<TInit, TCleanup>::get_instance()
{
    static std::weak_ptr<void> old_instance;

    if (std::shared_ptr<void> instance = old_instance.lock())
        return instance;

    init();
    std::shared_ptr<void> instance{nullptr, [](auto) {
                                       cleanup();
                                   }};
    old_instance = instance;
    return instance;
}

}  // namespace utils
