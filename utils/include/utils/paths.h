#pragma once

#include <filesystem>
#include <string_view>

namespace utils {

[[nodiscard]] std::filesystem::path get_kiwixz_home(std::string_view app_name);

}  // namespace utils
