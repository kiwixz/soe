cmake_minimum_required(VERSION 3.12)

set(target "utils")

file(GLOB_RECURSE src "*.cpp")
list(FILTER src EXCLUDE REGEX ".test.cpp$")

add_library("${target}" "${src}")
target_include_directories("${target}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (BUILD_TESTS)
    file(GLOB_RECURSE test_src "*.test.cpp")
    add_executable("${target}_test" "${test_src}")
    find_package("doctest" CONFIG REQUIRED)
    target_link_libraries("${target}_test" "${target}" "doctest::doctest")
    add_test(NAME "${target}_test" COMMAND "${target}_test")
endif ()


if (UNIX)
    find_package("Threads" REQUIRED)
    target_link_libraries("${target}" "Threads::Threads" "stdc++fs")
endif ()
