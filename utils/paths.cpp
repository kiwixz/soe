#include "utils/paths.h"
#include <utils/scope_exit.h>
#include <cstdlib>

#ifdef _WIN32
#    include <shlobj.h>
#else
#    include <pwd.h>
#    include <unistd.h>
#    include <vector>
#endif

namespace utils {

std::filesystem::path get_kiwixz_home(std::string_view app_name)
{
#ifdef _WIN32
    wchar_t* base;
    if (SHGetKnownFolderPath(FOLDERID_RoamingAppData, KF_FLAG_CREATE, nullptr, &base) != S_OK)
        throw std::runtime_error{"could not get appdata"};
    ScopeExit free_base{[&] {
        CoTaskMemFree(base);
    }};
    std::filesystem::path path = std::filesystem::path{base} / "kiwixz" / app_name;
#else
    const char* base;
    std::vector<char> buffer;
    if (!(base = std::getenv("HOME"))) {
        buffer.resize(4096);
        passwd pw;
        passwd* result;
        while (getpwuid_r(getuid(), &pw, buffer.data(), buffer.size(), &result) == ERANGE)
            buffer.resize(buffer.size() * 2);
        if (!result)
            throw std::runtime_error{"could not get HOME nor passwd of user"};
        base = result->pw_dir;  // pointee is in buffer
        if (!base)
            throw std::runtime_error{"user has no home"};
    }
    std::filesystem::path path = std::filesystem::path{base} / ".kiwixz" / app_name;
#endif
    std::filesystem::create_directories(path);
    return path;
}

}  // namespace utils
