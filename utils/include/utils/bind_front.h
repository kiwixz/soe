#pragma once

#include <functional>
#include <type_traits>
#include <utility>

namespace utils {

template <typename Function, typename... Args>
[[nodiscard]] auto bind_front(Function&& callable, Args&&... args);


template <typename Function, typename... Args>
auto bind_front(Function&& callable, Args&&... args)
{
    return [callable = std::forward<Function>(callable),
            bound_args_tuple = std::make_tuple(std::forward<Args>(args)...)](auto&&... call_args) -> decltype(auto) {  // decltype to support references
        return std::apply([&](auto&&... bound_args)
                          //noexcept(std::is_nothrow_invocable_v<Function, Args..., decltype(call_args)...>)  // propagate noexcept (not working?)
                          -> decltype(auto) {  // decltype to support references
                              return (std::invoke(callable,
                                                  std::forward<Args>(bound_args)...,
                                                  std::forward<decltype(call_args)>(call_args)...));
                          },
                          decltype(bound_args_tuple){bound_args_tuple});  // force a copy so we can call it multiple times
    };
}

}  // namespace utils
