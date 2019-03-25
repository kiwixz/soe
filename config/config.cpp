#include "config/config.h"
#include <fmt/printf.h>
#include <utils/paths.h>
#include <algorithm>
#include <fstream>
#include <vector>

namespace config {
namespace {

constexpr std::string_view implicit_value = "true";

}


bool Config::contains(std::string const& key) const
{
    return options_.count(key) > 0;
}

std::string const& Config::get_raw(std::string const& key) const
{
    return options_.at(key);
}

void Config::remove(std::string const& key)
{
    if (options_.erase(key) == 0)
        throw std::runtime_error{"key not present"};
}

void Config::clear()
{
    options_.clear();
}

bool Config::parse_args(int& argc, char** argv, bool allow_unknown)
{
    std::vector<int> used_args;
    used_args.reserve(argc - 1);

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            used_args.push_back(i);
            std::string_view arg = argv[i] + 1;
            if (arg == "-")
                break;
            if (arg == "-help")
                return true;

            size_t eq = arg.find('=');
            std::string key{arg.substr(0, eq)};
            std::string value;
            if (eq == std::string::npos)
                value = implicit_value;
            else
                value = std::string{arg.substr(eq + 1)};
            set_parsed_option(std::move(key), std::move(value), allow_unknown);
        }
        else if (argv[i][0] == '+') {
            used_args.push_back(i);
            parse_file(argv[i] + 1, allow_unknown);
        }
    }

    if (used_args.empty())
        return false;

    argc -= static_cast<int>(used_args.size());
    int old_index = used_args[0] + 1;
    int new_index = used_args[0];
    for (unsigned i = 1; i < used_args.size(); ++i) {
        for (; old_index < used_args[i]; ++old_index, ++new_index)
            argv[new_index] = argv[old_index];
        ++old_index;
    }
    for (; new_index <= argc; ++old_index, ++new_index)
        argv[new_index] = argv[old_index];

    return false;
}

void Config::parse_global_config(std::string_view app_name, bool allow_unknown)
{
    auto parse_if_exists = [&](std::filesystem::path const& path) {
        if (std::filesystem::exists(path))
            parse_file(path, allow_unknown);
    };

    constexpr std::string_view file_name = "config.ini";
    parse_if_exists(utils::get_kiwixz_home(app_name) / file_name);
    parse_if_exists(file_name);
}

void Config::parse_file(std::filesystem::path const& path, bool allow_unknown)
{
    std::ifstream ifs{path, std::ios::ate};
    if (!ifs)
        throw std::runtime_error{fmt::format("could not open config file '{}'", path)};
    std::streamoff size = ifs.tellg();
    std::string content(size, '\0');
    ifs.seekg(0, std::ios::beg);
    ifs.read(content.data(), size);
    ifs.close();
    parse_file_content(content, allow_unknown);
}

void Config::parse_file_content(std::string_view content, bool allow_unknown)
{
    char const* ptr = content.data();
    char const* end = content.data() + content.size();
    auto is_endline = [&](char const* c) {
        return c >= end || *c == '\n' || *c == '\r';
    };

    std::string prefix;
    while (ptr < end) {
        while (ptr < end && (is_endline(ptr) || *ptr == ' ' || *ptr == '\t'))
            ++ptr;
        if (ptr >= end)
            break;

        if (*ptr == '#') {
            while (!is_endline(ptr))
                ++ptr;
            continue;
        }
        else if (*ptr == '[') {
            ++ptr;
            char const* section_begin = ptr;
            while (!(is_endline(ptr) || *ptr == ']'))
                ++ptr;
            if (ptr >= end)
                break;
            char const* section_end = ptr;
            prefix = {section_begin, section_end};
            if (!prefix.empty())
                prefix += '.';
            ++ptr;
            continue;
        }

        char const* key_begin = ptr;
        while (!(is_endline(ptr) || *ptr == '='))
            ++ptr;
        char const* key_end = ptr;
        std::string key{key_begin, key_end};

        std::string value;
        if (ptr < end && *ptr == '=') {
            ++ptr;
            char const* value_begin = ptr;
            while (!is_endline(ptr))
                ++ptr;
            char const* value_end = ptr;
            value = {value_begin, value_end};
        }
        else
            value = implicit_value;

        set_parsed_option(prefix + std::move(key), std::move(value), allow_unknown);
    }
}

std::string Config::dump(std::string_view prefix) const
{
    std::vector<std::pair<std::string, std::string>> options(options_.begin(), options_.end());
    std::sort(options.begin(), options.end());
    std::string output;
    for (std::pair<std::string, std::string> const& option : options)
        output += fmt::format("{}{}={}\n", prefix, option.first, option.second);
    return output;
}

void Config::show_help(std::string_view app_name) const
{
    fmt::print("Usage: {} [--help] [+config_file] [-option...] [--] [positional arguments]\n", app_name);
    fmt::print("Options and their default values:\n{}", dump("\t"));
}

void Config::set_parsed_option(std::string key, std::string value, bool allow_unknown)
{
    auto it = options_.find(key);
    if (it == options_.end()) {
        if (allow_unknown)
            options_.insert({std::move(key), std::move(value)});
        else
            throw std::runtime_error{fmt::format("unknown option '{}'", key)};
    }
    else
        it->second = std::move(value);
}

}  // namespace config
