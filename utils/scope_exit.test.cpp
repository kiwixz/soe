#include "utils/scope_exit.h"
#include <doctest/doctest.h>

namespace utils::test {

TEST_SUITE("scope_exit")
{
    TEST_CASE("generic")
    {
        bool pass = false;
        {
            ScopeExitGeneric scope_exit{[&] {
                pass = true;
            }};
            CHECK(!pass);
        }
        CHECK(pass);
    }

    TEST_CASE("false")
    {
        ScopeExitGeneric{};
        ScopeExit<>{};
    }
}

}  // namespace utils::test
