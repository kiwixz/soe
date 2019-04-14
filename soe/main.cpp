#include "utils/config.h"
#include <opencv2/videoio.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <filesystem>
#include <string>

namespace soe {
namespace {

int parse_fourcc(std::string_view code)
{
    if (code.length() != 4)
        throw std::runtime_error{fmt::format("invalid fourcc '{}', length is not 4", code)};
    return cv::VideoWriter::fourcc(code[0], code[1], code[2], code[3]);
}

void main_impl(int argc, char** argv)
{
    utils::Config conf;
    conf.set("codec", "HFYU");

    conf.parse_global_config("soe");
    if (conf.parse_args(argc, argv) || argc != 3) {
        conf.show_help(argv[0], "input_file output_file");
        return;
    }
    std::string input_file = argv[1];
    std::string output_file = argv[2];

    cv::VideoCapture reader;
    if (!reader.open(input_file))
        throw std::runtime_error{fmt::format("could not open source video '{}' (codec/container may be unsupported)", input_file)};

    double out_fps = reader.get(cv::CAP_PROP_FPS);
    cv::Size frame_size{static_cast<int>(reader.get(cv::CAP_PROP_FRAME_WIDTH)),
                        static_cast<int>(reader.get(cv::CAP_PROP_FRAME_HEIGHT))};

    cv::VideoWriter writer;
    if (!writer.open(output_file, parse_fourcc(conf.get_raw("codec")), out_fps, frame_size))
        throw std::runtime_error{fmt::format("could not open destination video '{}' (codec/container may be unsupported)", output_file)};

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
