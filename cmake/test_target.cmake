function (test_cpp_target target test_src)
    if (BUILD_TESTS)
        list(APPEND test_src "${CMAKE_SOURCE_DIR}/cmake/main.test.cpp")
        add_executable("${target}_test" "${test_src}")
        find_package("doctest" CONFIG REQUIRED)
        target_link_libraries("${target}_test" "${target}" "doctest::doctest")
        add_test(NAME "${target}_test" COMMAND "${target}_test")
    endif ()
endfunction ()
