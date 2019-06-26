#include "soe/frame_stream.h"
#include "soe/frame_stream_cuda.h"
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
        throw utils::Exception{"invalid fourcc '{}', length is not 4", code};
    return cv::VideoWriter::fourcc(code[0], code[1], code[2], code[3]);
}

void main_impl(int argc, char** argv)
{
    utils::Config conf;

    conf.set("codec", "HFYU");
    conf.set("cuda", "true");
    conf.set("fps", "60.0");
    conf.set("speed", "1.0");

    conf.set("farneback.num_levels", "5");
    conf.set("farneback.pyr_scale", "0.5");
    conf.set("farneback.fast_pyramids", "false");
    conf.set("farneback.win_size", "13");
    conf.set("farneback.num_iters", "10");
    conf.set("farneback.poly_n", "5");
    conf.set("farneback.poly_sigma", "1.1");

    conf.parse_global_config("soe");
    if (conf.parse_args(argc, argv) || argc != 3) {
        conf.show_help(argv[0], "input_file output_file");
        return;
    }
    std::string input_file = argv[1];
    std::string output_file = argv[2];

    cv::VideoCapture reader;
    if (!reader.open(input_file))
        throw utils::Exception{"could not open source video '{}' (codec/container may be unsupported)", input_file};

    auto out_video_fps = conf.get<double>("fps");
    auto out_fps = out_video_fps / conf.get<double>("speed");
    cv::Size frame_size{static_cast<int>(reader.get(cv::CAP_PROP_FRAME_WIDTH)),
                        static_cast<int>(reader.get(cv::CAP_PROP_FRAME_HEIGHT))};

    cv::VideoWriter writer;
    if (!writer.open(output_file, parse_fourcc(conf.get_raw("codec")), out_video_fps, frame_size))
        throw utils::Exception{"could not open destination video '{}' (codec/container may be unsupported)", output_file};

    FarnebackSettings farneback_settings{conf.get<int>("farneback.num_levels"),
                                         conf.get<double>("farneback.pyr_scale"),
                                         conf.get<bool>("farneback.fast_pyramids"),
                                         conf.get<int>("farneback.win_size"),
                                         conf.get<int>("farneback.num_iters"),
                                         conf.get<int>("farneback.poly_n"),
                                         conf.get<double>("farneback.poly_sigma")};

    if (conf.get<bool>("cuda")) {
        FrameStreamCuda stream{out_fps, farneback_settings};
        cv::Mat frame_cpu;
        FrameStreamCuda::Frame frame;
        frame.timestamp = reader.get(cv::CAP_PROP_POS_MSEC) / 1000;
        while (reader.read(frame_cpu)) {
            fmt::print("Analyzing frame {}/{}...\r",
                       static_cast<int>(reader.get(cv::CAP_PROP_POS_FRAMES)) + 1,
                       static_cast<int>(reader.get(cv::CAP_PROP_FRAME_COUNT)));
            frame.picture = cv::cuda::GpuMat{frame_cpu};  // GpuMat is like a shared_ptr without move, so we must create another one
            stream.input_frame(std::move(frame));
            while (stream.has_output()) {
                FrameStreamCuda::Frame out_frame = stream.output_frame();
                out_frame.picture.download(frame_cpu);
                writer.write(frame_cpu);
            }
            frame.timestamp = reader.get(cv::CAP_PROP_POS_MSEC) / 1000;
        }
    }
    else {
        FrameStream stream{out_fps, farneback_settings};
        FrameStream::Frame frame;
        frame.timestamp = reader.get(cv::CAP_PROP_POS_MSEC) / 1000;
        while (reader.read(frame.picture)) {
            fmt::print("Analyzing frame {}/{}...\r",
                       static_cast<int>(reader.get(cv::CAP_PROP_POS_FRAMES)) + 1,
                       static_cast<int>(reader.get(cv::CAP_PROP_FRAME_COUNT)));
            stream.input_frame(std::move(frame));
            while (stream.has_output())
                writer.write(stream.output_frame().picture);
            frame.timestamp = reader.get(cv::CAP_PROP_POS_MSEC) / 1000;
        }
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
    catch (const std::exception& ex) {
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
