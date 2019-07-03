cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(CMAKE_CXX_COMPILER "clang++" CACHE FILEPATH "")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "")

project("soe" CUDA CXX)

option(BUILD_TESTS "Build tests" ON)
option(NATIVE_OPTIMIZATION "Perform optimizations specific to current machine" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Weverything  \
        -Wno-c++98-compat-pedantic -Wno-disabled-macro-expansion -Wno-double-promotion -Wno-exit-time-destructors  \
        -Wno-float-equal -Wno-padded -Wno-return-std-move-in-c++11 -Wno-sign-conversion -Wno-weak-vtables"
    )
    set(CMAKE_CXX_FLAGS_DEBUG "-D DEBUG -O1 -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS_RELEASE "-D NDEBUG -O3 -flto -ffast-math")
    link_libraries("-lpthread -lstdc++fs")
    if (NATIVE_OPTIMIZATION)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin clang -Xcompiler=\"${CMAKE_CXX_FLAGS}\"")
    set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=\"${CMAKE_CXX_FLAGS_DEBUG}\"")
    set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=\"${CMAKE_CXX_FLAGS_RELEASE}\"")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /nologo /MP /diagnostics:caret /W4 /permissive- /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "/D DEBUG /MTd /JMC /RTC1 /ZI")
    set(CMAKE_CXX_FLAGS_RELEASE "/D NDEBUG /MT /O2 /GL /Gw /Zi /fp:fast")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "/DEBUG:FASTLINK")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG:INCREMENTAL /DEBUG:FULL")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG")
    link_libraries("onecore.lib")
    if (NATIVE_OPTIMIZATION)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    endif ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"${CMAKE_CXX_FLAGS} /permissive\"")
    set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=\"${CMAKE_CXX_FLAGS_DEBUG}\"")
    set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=\"${CMAKE_CXX_FLAGS_RELEASE}\"")
endif ()

enable_testing()
find_package("Python3" COMPONENTS "Interpreter")
add_test(NAME "check_clang_format"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/check_clang_format.py"
)
add_test(NAME "check_clang_tidy"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/check_clang_tidy.py" "${CMAKE_BINARY_DIR}"
)


add_subdirectory("utils")

add_subdirectory("libsoe")
add_subdirectory("soe")
