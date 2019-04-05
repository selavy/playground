#include <catch2/catch.hpp>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <climits>

struct Bucketeer {
    Bucketeer(std::vector<uint32_t> bs) : total(0ull), buckets(std::move(bs)), counts(buckets.size() + 1, 0)
    {
        buckets.push_back(UINT32_MAX);
        assert(!buckets.empty() && "invalid buckets configuration");
    }

    void add(uint32_t x) noexcept {
        size_t bucket = _find_bucket(x);
        ++counts[bucket];
        total += x;
    }

    void del(uint32_t x) noexcept {
        size_t bucket = _find_bucket(x);
        --counts[bucket];
        total -= x;
    }

    size_t _find_bucket(uint32_t x) const noexcept {
        assert(x < UINT32_MAX && "invalid value");
        for (size_t i = 0; ; ++i) {
            if (x < buckets[i]) {
                return i;
            }
        }
    }

    uint64_t total;
    std::vector<uint32_t> buckets;
    std::vector<uint32_t> counts;
};

TEST_CASE("Bucketing", "[buckets]") {
    std::vector<uint32_t> buckets = { 5, 10, 15, 20, 25 };

    Bucketeer bb{buckets};
    for (size_t i = 0; i < bb.buckets.size(); ++i) {
        REQUIRE(bb.counts[i] == 0);
    }

    for (size_t i = 0; i < buckets.size(); ++i) {
        auto edge = buckets[i];
        bb.add(edge);
        REQUIRE(bb.counts[i+1] == 1);
        REQUIRE(bb.total == edge);
        bb.del(edge);
        REQUIRE(bb.counts[i+1] == 0);
        REQUIRE(bb.total == 0ull);
    }

    bb.add(3);
    bb.add(4);
    bb.add(3);
    REQUIRE(bb.counts[0] == 3u);
    REQUIRE(bb.total == 10ull);

    bb.add(5);
    REQUIRE(bb.counts[0] == 3u);
    REQUIRE(bb.counts[1] == 1u);
    REQUIRE(bb.total == 15ull);
}
