include(FindPackageHandleStandardArgs)

find_path(FFMPEG_INCLUDE_DIRS NAMES libavcodec/avcodec.h)
unset(FFMPEG_LIBRARIES)
foreach (FFMPEG_SUBLIBRARY avformat avdevice avcodec avutil swscale)
    find_library(FFMPEG_lib${FFMPEG_SUBLIBRARY}_LIBRARY NAMES ${FFMPEG_SUBLIBRARY})
    list(APPEND FFMPEG_LIBRARIES ${FFMPEG_lib${FFMPEG_SUBLIBRARY}_LIBRARY})
endforeach ()
if (WIN32)
    list(APPEND FFMPEG_LIBRARIES wsock32 ws2_32 Secur32)
endif ()

include_directories(SYSTEM ${FFMPEG_INCLUDE_DIRS})  # OpenCV 4.0.1 need this, seems fixed on master

find_package_handle_standard_args(FFMPEG REQUIRED_VARS FFMPEG_LIBRARIES FFMPEG_INCLUDE_DIRS)
