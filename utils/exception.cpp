#include "utils/exception.h"

namespace utils {

Exception::Exception(std::string_view what) :
    std::runtime_error{what.data()}
{}

}  // namespace utils
