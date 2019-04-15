#include <catch2/catch.hpp>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <errno.h>

double Sign(double x) noexcept
{
    if (x != x)
        return x;
    else if (x == 0.0)
        return 0.0;
    else
        return std::signbit(x) ? -1.0 : 1.0;
}

TEST_CASE("Floaty", "[floaty]")
{
    std::vector<std::tuple<int32_t, int32_t, bool>> cases = {
        { 0, -206, false },
        { 0, -228, false },
        { 0, -293, false },
        { 0, -151, false },
    };

    auto inv = [](double x) -> double
    {
        return Sign(x) * (std::exp(std::fabs(x)) - 1);
    };

    for (auto& cc : cases)
    {
        int32_t agg_size_chg = std::get<0>(cc);
        int32_t opp_size_chg = std::get<1>(cc);
        bool expected = std::get<2>(cc);
        double aggSizeChange = agg_size_chg;
        double oppSizeChange = opp_size_chg;

        double alo = 0.0;
        double ahi = 5.0;
        double blo = -INFINITY;
        double bhi = -5.0;

        auto GetNext = [](double x) -> double
        {
            if (x == INFINITY || x == -INFINITY)
                return x;
            return std::nextafter(x, INFINITY);
        };

        auto GetPrev = [](double x) -> double
        {
            if (x == INFINITY || x == -INFINITY)
                return x;
            return std::nextafter(x, -INFINITY);
        };

        double oppLo = inv(alo);
        double oppHi = inv(ahi);
        double aggLo = inv(blo);
        double aggHi = inv(bhi);

        oppLo = GetNext(oppLo);
        oppHi = GetPrev(oppHi);
        aggLo = GetNext(aggLo);
        aggHi = GetPrev(aggHi);

        bool result = (oppLo <= oppSizeChange) && (oppSizeChange <= oppHi) && (aggLo <= aggSizeChange) && (aggSizeChange <= aggHi);

        REQUIRE(result == expected);
    }
}

