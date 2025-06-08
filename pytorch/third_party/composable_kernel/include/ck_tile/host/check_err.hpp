// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/ranges.hpp"

namespace ck_tile {

template <typename ComputeDataType, typename OutDataType, typename AccDataType = ComputeDataType>
double get_relative_threshold(const int number_of_accumulations = 1)
{
    using F8   = ck_tile::fp8_t;
    using BF8  = ck_tile::bf8_t;
    using F16  = ck_tile::half_t;
    using BF16 = ck_tile::bf16_t;
    using F32  = float;
    using I8   = int8_t;
    using I32  = int32_t;

    static_assert(is_any_of<ComputeDataType, F8, BF8, F16, BF16, F32, I8, I32, int>::value,
                  "Warning: Unhandled ComputeDataType for setting up the relative threshold!");

    double compute_error = 0;
    if constexpr(is_any_of<ComputeDataType, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        compute_error = std::pow(2, -numeric_traits<ComputeDataType>::mant) * 0.5;
    }

    static_assert(is_any_of<OutDataType, F8, BF8, F16, BF16, F32, I8, I32, int>::value,
                  "Warning: Unhandled OutDataType for setting up the relative threshold!");

    double output_error = 0;
    if constexpr(is_any_of<OutDataType, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        output_error = std::pow(2, -numeric_traits<OutDataType>::mant) * 0.5;
    }
    double midway_error = std::max(compute_error, output_error);

    static_assert(is_any_of<AccDataType, F8, BF8, F16, BF16, F32, I8, I32, int>::value,
                  "Warning: Unhandled AccDataType for setting up the relative threshold!");

    double acc_error = 0;
    if constexpr(is_any_of<AccDataType, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        acc_error = std::pow(2, -numeric_traits<AccDataType>::mant) * 0.5 * number_of_accumulations;
    }
    return std::max(acc_error, midway_error);
}

template <typename ComputeDataType, typename OutDataType, typename AccDataType = ComputeDataType>
double get_absolute_threshold(const double max_possible_num, const int number_of_accumulations = 1)
{
    using F8   = ck_tile::fp8_t;
    using BF8  = ck_tile::bf8_t;
    using F16  = ck_tile::half_t;
    using BF16 = ck_tile::bf16_t;
    using F32  = float;
    using I8   = int8_t;
    using I32  = int32_t;

    static_assert(is_any_of<ComputeDataType, F8, BF8, F16, BF16, F32, I8, I32, int>::value,
                  "Warning: Unhandled ComputeDataType for setting up the absolute threshold!");

    auto expo            = std::log2(std::abs(max_possible_num));
    double compute_error = 0;
    if constexpr(is_any_of<ComputeDataType, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        compute_error = std::pow(2, expo - numeric_traits<ComputeDataType>::mant) * 0.5;
    }

    static_assert(is_any_of<OutDataType, F8, BF8, F16, BF16, F32, I8, I32, int>::value,
                  "Warning: Unhandled OutDataType for setting up the absolute threshold!");

    double output_error = 0;
    if constexpr(is_any_of<OutDataType, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        output_error = std::pow(2, expo - numeric_traits<OutDataType>::mant) * 0.5;
    }
    double midway_error = std::max(compute_error, output_error);

    static_assert(is_any_of<AccDataType, F8, BF8, F16, BF16, F32, I8, I32, int>::value,
                  "Warning: Unhandled AccDataType for setting up the absolute threshold!");

    double acc_error = 0;
    if constexpr(is_any_of<AccDataType, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        acc_error =
            std::pow(2, expo - numeric_traits<AccDataType>::mant) * 0.5 * number_of_accumulations;
    }
    return std::max(acc_error, midway_error);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    using size_type = typename std::vector<T>::size_type;

    os << "[";
    for(size_type idx = 0; idx < v.size(); ++idx)
    {
        if(0 < idx)
        {
            os << ", ";
        }
        os << v[idx];
    }
    return os << "]";
}

template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_floating_point_v<ranges::range_value_t<Range>> &&
        !std::is_same_v<ranges::range_value_t<Range>, half_t>,
    bool>::type CK_TILE_HOST
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg  = "Error: Incorrect results!",
          double rtol             = 1e-5,
          double atol             = 3e-6,
          bool allow_infinity_ref = false)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<double>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = *std::next(std::begin(out), i);
        const double r = *std::next(std::begin(ref), i);
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, bf16_t>,
    bool>::type CK_TILE_HOST
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg  = "Error: Incorrect results!",
          double rtol             = 1e-3,
          double atol             = 1e-3,
          bool allow_infinity_ref = false)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count = 0;
    double err    = 0;
    // TODO: This is a hack. We should have proper specialization for bf16_t data type.
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, half_t>,
    bool>::type CK_TILE_HOST
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg  = "Error: Incorrect results!",
          double rtol             = 1e-3,
          double atol             = 1e-3,
          bool allow_infinity_ref = false)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = static_cast<double>(std::numeric_limits<ranges::range_value_t<Range>>::min());
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_integral_v<ranges::range_value_t<Range>> &&
                  !std::is_same_v<ranges::range_value_t<Range>, bf16_t>)
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                     || std::is_same_v<ranges::range_value_t<Range>, int4_t>
#endif
                 ,
                 bool>
    CK_TILE_HOST check_err(const Range& out,
                           const RefRange& ref,
                           const std::string& msg = "Error: Incorrect results!",
                           double                 = 0,
                           double atol            = 0)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count   = 0;
    int64_t err     = 0;
    int64_t max_err = std::numeric_limits<int64_t>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const int64_t o = *std::next(std::begin(out), i);
        const int64_t r = *std::next(std::begin(ref), i);
        err             = std::abs(o - r);

        if(err > atol)
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << " out[" << i << "] != ref[" << i << "]: " << o << " != " << r
                          << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, fp8_t>),
                 bool>
    CK_TILE_HOST check_err(const Range& out,
                           const RefRange& ref,
                           const std::string& msg               = "Error: Incorrect results!",
                           unsigned max_rounding_point_distance = 1,
                           double atol                          = 1e-1,
                           bool allow_infinity_ref              = false)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    static const auto get_rounding_point_distance = [](fp8_t o, fp8_t r) -> unsigned {
        static const auto get_sign_bit = [](fp8_t v) -> bool {
            return 0x80 & bit_cast<uint8_t>(v);
        };

        if(get_sign_bit(o) ^ get_sign_bit(r))
        {
            return std::numeric_limits<unsigned>::max();
        }
        else
        {
            return std::abs(bit_cast<int8_t>(o) - bit_cast<int8_t>(r));
        }
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const fp8_t o_fp8   = *std::next(std::begin(out), i);
        const fp8_t r_fp8   = *std::next(std::begin(ref), i);
        const double o_fp64 = type_convert<float>(o_fp8);
        const double r_fp64 = type_convert<float>(r_fp8);
        err                 = std::abs(o_fp64 - r_fp64);
        if(!(less_equal<double>{}(err, atol) ||
             get_rounding_point_distance(o_fp8, r_fp8) <= max_rounding_point_distance) ||
           is_infinity_error(o_fp64, r_fp64))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o_fp64 << " != " << r_fp64 << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, bf8_t>),
                 bool>
    CK_TILE_HOST check_err(const Range& out,
                           const RefRange& ref,
                           const std::string& msg  = "Error: Incorrect results!",
                           double rtol             = 1e-3,
                           double atol             = 1e-3,
                           bool allow_infinity_ref = false)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

} // namespace ck_tile
