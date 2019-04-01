include(vcpkg_common_functions)

vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO opencv/opencv
    REF 4.0.1
    SHA512 d74dd325029a67676dd2c7fdbecb2b14cb531441b2a1b74fc1ebe6db096ea87c12801c8b997ebbe280fbd401311c0220133a0c29911b0dae82cb93453be0b2b0
    HEAD_REF master
    PATCHES
        "${CMAKE_CURRENT_LIST_DIR}/0001-fix-paths.patch"
        "${CMAKE_CURRENT_LIST_DIR}/0002-fix-paths-linux.patch"
)

vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}
    PREFER_NINJA
    OPTIONS
        -DBUILD_LIST=videoio
        -DWITH_JASPER=OFF
        -DWITH_JPEG=OFF
        -DWITH_OPENEXR=OFF
        -DWITH_PNG=OFF
        -DWITH_PROTOBUF=OFF
        -DWITH_TIFF=OFF
        -DWITH_WEBP=OFF

        -DBUILD_opencv_apps=OFF
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_TESTS=OFF
        -DWITH_IPP=ON  # force it so debug has same number of libs

        # provided by vcpkg
        -DBUILD_JASPER=OFF
        -DBUILD_JPEG=OFF
        -DBUILD_OPENEXR=OFF
        -DBUILD_PNG=OFF
        -DBUILD_PROTOBUF=OFF
        -DBUILD_TIFF=OFF
        -DBUILD_WEBP=OFF
        -DBUILD_ZLIB=OFF
)

vcpkg_install_cmake()

if (WIN32)
    vcpkg_fixup_cmake_targets(CONFIG_PATH staticlib)

    file(REMOVE_RECURSE
        ${CURRENT_PACKAGES_DIR}/debug/include
        ${CURRENT_PACKAGES_DIR}/debug/share
        ${CURRENT_PACKAGES_DIR}/etc
        ${CURRENT_PACKAGES_DIR}/debug/etc
    )
    file(REMOVE
        ${CURRENT_PACKAGES_DIR}/LICENSE
        ${CURRENT_PACKAGES_DIR}/debug/LICENSE
        ${CURRENT_PACKAGES_DIR}/setup_vars_opencv4.cmd
        ${CURRENT_PACKAGES_DIR}/debug/setup_vars_opencv4.cmd
        ${CURRENT_PACKAGES_DIR}/OpenCVConfig.cmake
        ${CURRENT_PACKAGES_DIR}/debug/OpenCVConfig.cmake
        ${CURRENT_PACKAGES_DIR}/OpenCVConfig-version.cmake
        ${CURRENT_PACKAGES_DIR}/debug/OpenCVConfig-version.cmake
    )

    file(READ ${CURRENT_PACKAGES_DIR}/share/opencv/OpenCVConfig.cmake OPENCV_CONFIG)
    string(REPLACE
        "\${OpenCV_CONFIG_PATH}/../"
        "\${_VCPKG_INSTALLED_DIR}/\${VCPKG_TARGET_TRIPLET}/"
        OPENCV_CONFIG "${OPENCV_CONFIG}")
    file(WRITE ${CURRENT_PACKAGES_DIR}/share/opencv/OpenCVConfig.cmake "${OPENCV_CONFIG}")
else ()
    file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/share)
    vcpkg_fixup_cmake_targets(CONFIG_PATH lib/cmake/opencv4)

    file(REMOVE_RECURSE
        ${CURRENT_PACKAGES_DIR}/debug/include
        ${CURRENT_PACKAGES_DIR}/debug/share
        ${CURRENT_PACKAGES_DIR}/bin
        ${CURRENT_PACKAGES_DIR}/debug/bin
    )

    file(READ ${CURRENT_PACKAGES_DIR}/share/opencv/OpenCVConfig.cmake OPENCV_CONFIG)
    string(REPLACE
        "\${OpenCV_CONFIG_PATH}/../../../"
        "\${_VCPKG_INSTALLED_DIR}/\${VCPKG_TARGET_TRIPLET}/"
        OPENCV_CONFIG "${OPENCV_CONFIG}")
    file(WRITE ${CURRENT_PACKAGES_DIR}/share/opencv/OpenCVConfig.cmake "${OPENCV_CONFIG}")
endif ()

configure_file(${SOURCE_PATH}/LICENSE ${CURRENT_PACKAGES_DIR}/share/opencv/copyright COPYONLY)

if (WIN32)
    # lie to allow ffmpeg dlls
    set(VCPKG_LIBRARY_LINKAGE dynamic)
    set(VCPKG_POLICY_ALLOW_OBSOLETE_MSVCRT enabled)
endif ()
