#include <catch2/catch.hpp>
#include <vector>
#include <smmintrin.h>
#include <immintrin.h>
#include <cstdlib>
#include <cstring>

double dot1(const double* __restrict cs, const double* __restrict vs, size_t length) noexcept
{
    double result = 0.0;
    for (size_t i = 0; i < length; ++i)
    {
        result += cs[i]*vs[i];
    }
    return result;
}

double dot2(const double* __restrict cs, const double* __restrict vs, size_t length) noexcept
{
    double result = 0.0;
    size_t i = 0;
    size_t repeat = length / 8;
    size_t left   = length % 8;
    for (size_t j = 0; j < repeat; ++j)
    {
        result += cs[i+0]*vs[i+0];
        result += cs[i+1]*vs[i+1];
        result += cs[i+2]*vs[i+2];
        result += cs[i+3]*vs[i+3];
        result += cs[i+4]*vs[i+4];
        result += cs[i+5]*vs[i+5];
        result += cs[i+6]*vs[i+6];
        result += cs[i+7]*vs[i+7];
        i += 8;
    }
    switch(left)
    {
        case 7: result += cs[i+6]*vs[i+6];
        case 6: result += cs[i+5]*vs[i+5];
        case 5: result += cs[i+4]*vs[i+4];
        case 4: result += cs[i+3]*vs[i+3];
        case 3: result += cs[i+2]*vs[i+2];
        case 2: result += cs[i+1]*vs[i+1];
        case 1: result += cs[i+0]*vs[i+0];
        case 0: break;
    }
    // for (size_t j = 0; j < left; ++j)
    // {
    //     result += cs[i]*vs[i];
    //     i += 1;
    // }
    return result;
}

struct Model
{
    Model(std::vector<double> coef, std::vector<double> vals)
        : Model(std::vector<float>{coef.begin(), coef.end()}, std::vector<float>{vals.begin(), vals.end()})
    {}

    Model(std::vector<float> coef, std::vector<float> vals)
    {
        assert(coef.size() == vals.size());

        // force len to be a multiple of 8
        len = coef.size();
        if (len % 8 != 0) {
            len += 8 - (len % 8);
        }
        assert(len % 8 == 0);

        size_t alignment = 32;
        if (posix_memalign((void**)&cs, alignment, len * sizeof(float)) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        if (posix_memalign((void**)&vs, alignment, len * sizeof(float)) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }

        memcpy(&cs[0], coef.data(), coef.size()*sizeof(float));
        memcpy(&vs[0], vals.data(), vals.size()*sizeof(float));
        memset(&cs[coef.size()], 0, (len - coef.size())*sizeof(float));
        memset(&vs[vals.size()], 0, (len - vals.size())*sizeof(float));

#if 0
        printf("Calculating:\n");
        double result = 0;
        for (size_t i = 0; i < len; ++i) {
            printf("A[%zu] * B[%zu] = %f * %f = %f\n", i, i, cs[i], vs[i], cs[i]*vs[i]);
            result += cs[i]*vs[i];
        }
        printf("Result = %f\n", result);
#endif
    }

    ~Model() { free(cs); free(vs); }

    double eval() const noexcept
    {
        assert(len % 8 == 0);
#if 0
        double result = 0.;
        alignas(32) float tmp[8];
        for (size_t i = 0; i < len; i += 8)
        {
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i], *(__m256*)&vs[i], /*imm8*/0xFF));
            result += tmp[0];
            result += tmp[4];
        }
        return result;
#endif

        double result = 0.;
        size_t groups = len / 8;
        size_t repeat = groups / 4;
        size_t left   = groups % 4;
        alignas(32) float tmp[8];
        size_t i = 0;
        for (size_t j = 0; j < repeat; ++j)
        {
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i+8*0], *(__m256*)&vs[i+8*0], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i+8*1], *(__m256*)&vs[i+8*1], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i+8*2], *(__m256*)&vs[i+8*2], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i+8*3], *(__m256*)&vs[i+8*3], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            i += 8*4;
        }
        for (size_t j = 0; j < left; ++j)
        {
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i], *(__m256*)&vs[i], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            i += 8;
        }
        return result;
    }

    float* __restrict cs;
    float* __restrict vs;
    size_t len;
};

#if 0
struct DblModel
{
    DblModel(std::vector<double> coef, std::vector<double> vals)
    {
        assert(coef.size() == vals.size());

        // force len to be a multiple of 8
        len = coef.size();
        if (len % 8 != 0) {
            len += 8 - (len % 8);
        }
        assert(len % 8 == 0);

        size_t alignment = 32;
        if (posix_memalign((void**)&cs, alignment, len * sizeof(float)) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        if (posix_memalign((void**)&vs, alignment, len * sizeof(float)) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }

        memcpy(&cs[0], coef.data(), coef.size()*sizeof(cs[0]));
        memcpy(&vs[0], vals.data(), vals.size()*sizeof(vs[0]));
        memset(&cs[coef.size()], 0, (len - coef.size())*sizeof(cs[0]));
        memset(&vs[vals.size()], 0, (len - vals.size())*sizeof(vs[0]));
    }

    ~DblModel() { free(cs); free(vs); }

    double eval() const noexcept
    {
        assert(len % 8 == 0);
        double result = 0.;
        size_t groups = len / 8;
        size_t repeat = groups / 4;
        size_t left   = groups % 4;
        alignas(32) float tmp[8];
        size_t i = 0;
        for (size_t j = 0; j < repeat; ++j)
        {
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i+8*0], *(__m256*)&vs[i+8*0], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i+8*1], *(__m256*)&vs[i+8*1], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i+8*2], *(__m256*)&vs[i+8*2], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i+8*3], *(__m256*)&vs[i+8*3], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            i += 8*4;
        }
        for (size_t j = 0; j < left; ++j)
        {
            _mm256_store_ps(&tmp[0], _mm256_dp_ps(*(__m256*)&cs[i], *(__m256*)&vs[i], /*imm8*/0xFF));
            result += tmp[0] + tmp[4];
            i += 8;
        }
        return result;
    }

    double* cs;
    double* vs;
    size_t len;
};
#endif

TEST_CASE("DotProduct", "[unroll]")
{
    SECTION("Odd number of elements")
    {
        std::vector<double> As = { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13. };
        std::vector<double> Bs = { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13. };
        double expect = 819;

        REQUIRE(dot1(As.data(), Bs.data(), As.size()) == expect);
        REQUIRE(dot2(As.data(), Bs.data(), As.size()) == expect);

        REQUIRE(Model{As, Bs}.eval() == expect);
    }

    SECTION("Small")
    {
        std::vector<double> As = {
             12.,  65.,  18.,  83.,  54., 
        };
        std::vector<double> Bs = {
              1.,  84.,  37.,  70.,  74., 
        };
        double expect = 15944;

        REQUIRE(dot1(As.data(), Bs.data(), As.size()) == expect);
        REQUIRE(dot2(As.data(), Bs.data(), As.size()) == expect);
        REQUIRE(Model{As, Bs}.eval() == expect);
    }

    SECTION("Medium")
    {
        std::vector<double> As = {
             68.,  34.,  19.,  86.,  30.,  47.,  18.,  35., 
             21.,  33.,  35.,  84.,  52.,  16.,  51.,  68., 
             73.,  63.,  52.,  50.,  64.,  76.,  95.,  24., 
             72.,  78.,  93.,  23.,  65.,  69.,  20.,  57., 
             98.,   1.,  14.,  46.,  80.,  18., 
        };
        std::vector<double> Bs = {
             90.,  81.,  91.,  88.,  66.,  18.,  44.,  49., 
             93.,  11.,  11.,  71.,  87.,  25.,   8.,  98., 
             16.,  49.,  99.,  30.,  64.,  62.,  77.,  37., 
             80.,  57.,  42.,  78.,  40.,  88.,  65.,  12., 
             25.,  36.,  75.,  38.,  96.,  21., 
        };
        double expect = 111983;

        REQUIRE(dot1(As.data(), Bs.data(), As.size()) == expect);
        REQUIRE(dot2(As.data(), Bs.data(), As.size()) == expect);
        REQUIRE(Model{As, Bs}.eval() == expect);
    }

    SECTION("Larger")
    {
        std::vector<double> As = {
             65.,  60.,  99.,  68.,  30.,   9.,  32.,  48., 
             91.,  83.,  20.,  95.,  37.,  29.,  42.,  12., 
              1.,  32.,  23.,  11.,  90.,  48.,  57.,  47., 
             48.,  49.,   6.,  33.,  53.,  46.,  32.,  77., 
             73.,   3.,  16.,  95.,  97.,  20.,  79.,  34., 
             51.,  24.,  63.,  10.,   5.,  50.,  85.,  99., 
             18.,   8.,  22.,  22.,  54.,   9.,  89.,  61., 
             43.,  41.,  35.,  42.,  39.,  48.,  89.,  67., 
             78.,  93.,  68.,  41.,  79.,  99.,  25.,  69., 
             18.,  83.,  11.,  13.,  46.,  92.,  29.,  96., 
              6.,  38.,  55.,  98.,  70.,  96.,  95.,  74., 
              7.,  83.,  84.,  90.,  79.,  92.,  22.,  63., 
             11.,  11.,  50.,  91.,  69., 
        };
        std::vector<double> Bs = {
             43.,  87.,  75.,  24.,  86.,  12.,  17.,  32., 
              1.,  75.,   6.,  85.,  12.,  58.,  55.,  39., 
             21.,  32.,  58.,  54.,  79.,  49.,   2.,  55., 
             91.,  83.,  76.,  12.,  47.,  71.,  70.,  39., 
             74.,  20.,  38.,  33.,  91.,  33.,  47.,  15., 
             38.,   8.,  39.,  62.,  13.,  38.,  18.,   1., 
             14.,   2.,  36.,  22.,  76.,  14.,  36.,  75., 
             96.,  81.,  28.,  77.,  16.,  31.,  31.,  68., 
             10.,  90.,  28.,  74.,   3.,  35.,  58.,  64., 
              7.,  57.,   8.,  81.,  51.,  34.,  24.,  17., 
             47.,   2.,  72.,  20.,  16.,  38.,  12.,  97., 
             40.,  40.,  34.,  37.,  46.,  54.,  31.,  67., 
             63.,  32.,  43.,  65.,  86., 
        };
        double expect = 238190;

        REQUIRE(dot1(As.data(), Bs.data(), As.size()) == expect);
        REQUIRE(dot2(As.data(), Bs.data(), As.size()) == expect);
        REQUIRE(Model{As, Bs}.eval() == expect);
    }
}

TEST_CASE("SIMD float", "[unroll]")
{
    float vs[4] = { 1.0, 2.0, 3.0, 4.0 };
    float cs[4] = { 1.0, 2.0, 3.0, 4.0 };
    float rs[4] = { 1.0, 1.0, 1.0, 1.0 };
    _mm_store_ps(&rs[0], _mm_dp_ps(*(__m128*)&vs[0], *(__m128*)&cs[0], /*imm8*/0xFF));
    for (int i = 0; i < 4; ++i) {
        printf("r[%d] = %f\n", i, rs[i]);
    }
}

TEST_CASE("SIMD double", "[unroll]")
{
    double vs[2] = { 3.0, 4.0 };
    double cs[2] = { 3.0, 4.0 };
    double rs[2] = { 1.0, 1.0 };
    _mm_store_pd(&rs[0], _mm_dp_pd(*(__m128d*)&vs[0], *(__m128d*)&cs[0], /*imm8*/0xFF));
    for (int i = 0; i < 2; ++i) {
        printf("r[%d] = %f\n", i, rs[i]);
    }
}

TEST_CASE("_mm256_dp_ps", "[unroll]")
{
    alignas(32) float vs[8] = { 1., 2., 3., 4., 5., 6., 7., 8. };
    alignas(32) float cs[8] = { 1., 2., 3., 4., 5., 6., 7., 8. };
    alignas(32) float rs[8];
    _mm256_store_ps(&rs[0], _mm256_dp_ps(*(__m256*)&vs[0], *(__m256*)&cs[0], /*imm8*/0xFF));
    for (int i = 0; i < 8; ++i) {
        printf("r[%d] = %f\n", i, rs[i]);
    }
    double result = rs[0] + rs[4];
    printf("result = %f\n", result);
}

