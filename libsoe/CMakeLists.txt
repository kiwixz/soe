cmake_minimum_required(VERSION 3.12)
include("${CMAKE_SOURCE_DIR}/cmake/test_target.cmake")

set(target "libsoe")

file(GLOB_RECURSE src "*.cpp")
list(FILTER src EXCLUDE REGEX ".test.cpp$")
file(GLOB_RECURSE src_cuda "*.cu")
list (APPEND src "${src_cuda}")

add_library("${target}" "${src}")
target_include_directories("${target}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE test_src "*.test.cpp")
test_cpp_target("${target}" "${test_src}")


find_package("OpenCV" CONFIG REQUIRED "cudaimgproc" "cudaoptflow" "cudawarping" "video")

target_link_libraries("${target}"
    "utils"

    "${OpenCV_LIBS}"
)
