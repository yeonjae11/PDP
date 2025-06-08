//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <string_view>
#include <tuple>

#include "matrix_portion.hpp"

// clang-format off
#define UKERNEL_MATMUL_VARIANT(name)          \
    {kai_get_m_step_matmul_## name,            \
     kai_get_n_step_matmul_## name,            \
     kai_get_mr_matmul_## name,                \
     kai_get_nr_matmul_## name,                \
     kai_get_kr_matmul_## name,                \
     kai_get_sr_matmul_## name,                \
     kai_get_lhs_packed_offset_matmul_## name, \
     kai_get_rhs_packed_offset_matmul_## name, \
     kai_get_dst_offset_matmul_## name,        \
     kai_get_dst_size_matmul_## name,          \
     kai_run_matmul_## name}

#define UKERNEL_MATMUL_PACK_VARIANT(name, features_check, lhs_pack, rhs_pack)                           \
    {                                                                                                   \
        {UKERNEL_MATMUL_VARIANT(name), "kai_matmul_" #name, features_check},                            \
        {                                                                                               \
            kai_get_lhs_packed_size_##lhs_pack,                                                         \
            kai_get_rhs_packed_size_##rhs_pack,                                                         \
            kai_get_lhs_packed_offset_##lhs_pack,                                                       \
            kai_get_rhs_packed_offset_##rhs_pack,                                                       \
            kai_get_lhs_offset_##lhs_pack,                                                              \
            kai_get_rhs_offset_##rhs_pack,                                                              \
            kai_run_##lhs_pack,                                                                         \
            kai_run_##rhs_pack                                                                          \
        }                                                                                               \
    }
// clang-format on

namespace kai::test {

template <typename T>
struct UkernelVariant {
    /// Interface for testing variant.
    T interface;

    /// Name of the test variant.
    std::string_view name{};

    /// Check if CPU supports required features.
    ///
    /// @return Supported (true) or not supported (false).
    std::function<bool(void)> fn_is_supported;

    UkernelVariant(T interface, const std::string_view name, const std::function<bool(void)>& fn_is_supported) :
        interface(interface), name(name), fn_is_supported(fn_is_supported) {
    }
};

template <typename T, typename P>
struct UkernelPackVariant {
    /// Interface for testing variant.
    UkernelVariant<T> ukernel;
    P pack_interface;
};
/// Matrix multiplication shape.
struct MatMulShape {
    size_t m{};  ///< LHS height.
    size_t n{};  ///< RHS width.
    size_t k{};  ///< LHS width and RHS height.

    struct Hash {
        size_t operator()(const kai::test::MatMulShape& shape) const {
            return                                     //
                (std::hash<size_t>{}(shape.m) << 0) ^  //
                (std::hash<size_t>{}(shape.n) << 1) ^  //
                (std::hash<size_t>{}(shape.k) << 2);
        }
    };

private:
    friend bool operator==(const MatMulShape& lhs, const MatMulShape& rhs) {
        return                 //
            lhs.m == rhs.m &&  //
            lhs.n == rhs.n &&  //
            lhs.k == rhs.k;
    }
};

/// Matrix multiplication test information.
using MatMulTestParams = std::tuple<size_t, MatMulShape>;
using MatMulTestPortionedParams = std::tuple<size_t, MatMulShape, MatrixPortion>;
using MatMulTestPortionedParamsWithBias = std::tuple<size_t, MatMulShape, MatrixPortion, bool>;

class UkernelVariantTest : public ::testing::TestWithParam<MatMulTestParams> {};

}  // namespace kai::test
