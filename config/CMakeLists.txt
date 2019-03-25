cmake_minimum_required(VERSION 3.12)
include("${CMAKE_SOURCE_DIR}/cmake/test_target.cmake")

set(target "config")

file(GLOB_RECURSE src "*.cpp")
list(FILTER src EXCLUDE REGEX ".test.cpp$")

add_library("${target}" "${src}")
target_include_directories("${target}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE test_src "*.test.cpp")
test_cpp_target("${target}" "${test_src}")


find_package("fmt" CONFIG REQUIRED)

target_link_libraries("${target}"
    "utils"

    "fmt::fmt"
)

if (UNIX)
    find_package("Threads" REQUIRED)
    target_link_libraries("${target}" "Threads::Threads" "stdc++fs")
endif ()
