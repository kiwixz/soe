cmake_minimum_required(VERSION 3.12)
include("${CMAKE_SOURCE_DIR}/cmake/test_target.cmake")

set(target "soe")

file(GLOB_RECURSE src "*.cpp")
list(REMOVE_ITEM src "main.cpp")
list(FILTER src EXCLUDE REGEX ".test.cpp$")

add_library("${target}_base" "${src}")
add_executable("${target}" "main.cpp")
target_link_libraries("${target}" "${target}_base")

file(GLOB_RECURSE test_src "*.test.cpp")
test_cpp_target("${target}_base" "${test_src}")


find_package("fmt" CONFIG REQUIRED)
find_package("spdlog" CONFIG REQUIRED)

target_link_libraries("${target}_base"
    "config"
    "libsoe"
    "utils"

    "fmt::fmt"
    "spdlog::spdlog"
)
