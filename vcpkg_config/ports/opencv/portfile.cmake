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
)

vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}
    PREFER_NINJA
    OPTIONS
        -DBUILD_LIST=core
        #-DBUILD_LIST=videoio
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

configure_file(${SOURCE_PATH}/LICENSE ${CURRENT_PACKAGES_DIR}/share/opencv/copyright COPYONLY)
