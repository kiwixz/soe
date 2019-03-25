#pragma once

#include <fmt/format.h>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace config {

struct Config {
    bool contains(std::string const& key) const;
    std::string const& get_raw(std::string const& key) const;

    void remove(std::string const& key);
    void clear();

    /// Parse command-line arguments (and config file if explicitly given).
    /// Parsed arguments will be removed.
    /// Return true if application should show help (all arguments may not be parsed, but you can call again).
    bool parse_args(int& argc, char** argv, bool allow_unknown = false);

    void parse_global_config(std::string_view app_name, bool allow_unknown = false);
    void parse_file(std::filesystem::path const& path, bool allow_unknown = false);
    void parse_file_content(std::string_view content, bool allow_unknown = false);
    std::string dump(std::string_view prefix) const;
    void show_help(std::string_view app_name) const;

    template <typename T>
    T get(std::string const& key) const;

    template <typename T>
    void get(std::string const& key, T& value) const;

    template <typename T>
    void set(std::string key, T&& value);

private:
    std::unordered_map<std::string, std::string> options_;

    void set_parsed_option(std::string key, std::string value, bool allow_unknown);
};


template <typename T>
T Config::get(std::string const& key) const
{
    std::string const& value = get_raw(key);
    if constexpr (std::is_same_v<T, bool>) {
        if (value == "true" || value == "1")
            return true;
        else if (value == "false" || value == "0")
            return false;
        throw std::runtime_error{fmt::format("key '{}': expected boolean value, got '{}'", key, value)};
    }
    else if constexpr (std::is_arithmetic_v<T>) {
        if (value.length() == 0)
            throw std::runtime_error{fmt::format("key '{}': expected numeric value, got empty string", key)};

        char* end;
        T result;
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_unsigned_v<T>)
                result = static_cast<T>(std::strtoull(value.c_str(), &end, 0));
            else
                result = static_cast<T>(std::strtoll(value.c_str(), &end, 0));
        }
        else
            result = static_cast<T>(std::strtold(value.c_str(), &end));

        if (end != value.c_str() + value.length())
            throw std::runtime_error{fmt::format("key '{}': expected {} value, got '{}'",
                                                 key, std::is_integral_v<T> ? "integer" : "floating-point", value)};
        return result;
    }
    else
        return T{value};
}

template <typename T>
void Config::get(std::string const& key, T& value) const
{
    value = get<T>(key);
}

template <typename T>
void Config::set(std::string key, T&& value)
{
    if constexpr (std::is_same_v<T, bool>)
        return set(std::move(key), value ? "true" : "false");
    else if constexpr (std::is_arithmetic_v<T>)
        return set(std::move(key), std::to_string(value));
    else
        options_.insert_or_assign(std::move(key), std::forward<T>(value));
}

}  // namespace config
