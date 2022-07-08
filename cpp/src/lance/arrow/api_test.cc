/// Test the public header does not leak internal implementations.
///
/// We should only include headers under `/cpp/include`.
///
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Read full table") {}