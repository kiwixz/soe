include(vcpkg_common_functions)

vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO opencv/opencv
    REF 4.1.0
    SHA512 492168c1260cd30449393c4b266d75202e751493a8f1e184af6c085d8f4a38800ee954d84fe8c36fcceb690b1ebb5e511b68c05901f64be79a0915f3f8a46dc0
    HEAD_REF master
    PATCHES
        "${CMAKE_CURRENT_LIST_DIR}/0001-fix-paths.patch"
        "${CMAKE_CURRENT_LIST_DIR}/0002-fix-paths-linux.patch"
        "${CMAKE_CURRENT_LIST_DIR}/0003-fix-compilation.patch"
)

vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}
    PREFER_NINJA
    OPTIONS
        -DBUILD_LIST=video,videoio
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

        -DOPENCV_FFMPEG_USE_FIND_PACKAGE=ON
        -DFFMPEG_DIR=${CMAKE_CURRENT_LIST_DIR}
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
