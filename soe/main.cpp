#include <opencv2/videoio.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <filesystem>
#include <string>

namespace soe {
namespace {

void main_impl(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    cv::VideoCapture reader;
    if (!reader.open(argv[1]))
        throw std::runtime_error{fmt::format("could not open source video '{}'", argv[1])};

    double out_fps = reader.get(cv::CAP_PROP_FPS);  // may return garbage
    cv::Size frame_size{static_cast<int>(reader.get(cv::CAP_PROP_FRAME_WIDTH)),
                        static_cast<int>(reader.get(cv::CAP_PROP_FRAME_HEIGHT))};

    cv::VideoWriter writer;
    if (!writer.open(argv[2], cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), out_fps, frame_size))
        throw std::runtime_error{fmt::format("could not open destination video '{}'", argv[2])};

    cv::Mat frame;
    while (reader.read(frame)) {
        writer.write(frame);
    }
}

int main(int argc, char** argv)
{
    try {
        std::string log_file_path = (std::filesystem::temp_directory_path() / "soe.log").string();
        auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path, true);
        auto logger = std::make_shared<spdlog::logger>("", spdlog::sinks_init_list{console_sink, file_sink});
        spdlog::set_default_logger(logger);
#ifdef DEBUG
        spdlog::set_level(spdlog::level::debug);
#endif
        spdlog::set_pattern("%^[%H:%M:%S.%f][%t][%l]%$ %v");
        main_impl(argc, argv);
        spdlog::drop_all();
    }
    catch (std::exception const& ex) {
        spdlog::critical(ex.what());
        return 1;
    }
    return 0;
}

}  // namespace
}  // namespace soe


int main(int argc, char** argv)
{
    soe::main(argc, argv);
}
