//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp_qsi4cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_format.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/pad.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {

static auto cpu_has_dotprod_and_fp16 = []() { return cpu_has_dotprod() && cpu_has_fp16(); };
static auto cpu_has_i8mm_and_fp16 = []() { return cpu_has_i8mm() && cpu_has_fp16(); };

static const std::array<UkernelVariant<kai_matmul_clamp_f16_qai8dxp_qsi4cxp_ukernel>, 4>
    variants_kai_matmul_clamp_f16_qai8dxp_qsi4cxp = {{
        {UKERNEL_MATMUL_VARIANT(clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod),
         "kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod", cpu_has_dotprod_and_fp16},
        {UKERNEL_MATMUL_VARIANT(clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod),
         "kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod", cpu_has_dotprod_and_fp16},
        {UKERNEL_MATMUL_VARIANT(clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod),
         "kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod", cpu_has_dotprod_and_fp16},
        {UKERNEL_MATMUL_VARIANT(clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm),
         "kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm", cpu_has_i8mm_and_fp16},
    }};

class MatMulTest_f16_qai8dxp_qsi4cxp : public ::testing::TestWithParam<MatMulTestPortionedParamsWithBias> {};

TEST_P(MatMulTest_f16_qai8dxp_qsi4cxp, EndToEnd) {
    const auto& [variant_index, matmul_shape, portion, has_bias] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f16_qai8dxp_qsi4cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "CPU features are not supported by current CPU";
    }

    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    if (mr == 1 && M > 1) {
        GTEST_SKIP() << "Kernel does not support M != 1";
    }

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    // Generates input data.
    const auto ref_lhs_f16 = fill_random<Float16>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    std::vector<uint8_t> ref_biases;

    if (has_bias) {
        ref_biases = fill_random<float>(N, seed + 2);
    }
    // For reference implementation, Casting FP16 input to FP32 type and FP32 output back to FP16 because the matmul
    // implementation works with FP32 accumulation and casts the result to FP16
    const auto ref_lhs = cast<float, Float16>(ref_lhs_f16.data(), ref_lhs_f16.size() * 8 / size_in_bits<Float16>);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit symmetric quantization.
    //   * Quantizes the RHS matrix using 8-bit asymmetric quantization.
    //   * Performs GEMM.
    const auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    const auto [ref_rhs_qsi4, ref_rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, Int4, float>(ref_rhs.data(), N, K, K);

    const auto ref_dst_no_clamp =
        matmul_nt_t_quantized<int8_t, float, int32_t, Int4, float, int32_t, float, float, int32_t, float>(
            M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), 1, K,
            ref_rhs_qsi4.data(), ref_rhs_scales.data(), nullptr, 1, K, has_bias ? ref_biases.data() : nullptr, nullptr,
            nullptr, 1);

    // Clamps the reference output.
    const auto clamp_ratio = 0.8F;
    const auto [clamp_min, clamp_max] = find_clamp_range<float>(ref_dst_no_clamp.data(), M * N, clamp_ratio);
    const auto ref_dst_float = clamp<float>(ref_dst_no_clamp.data(), M * N, clamp_min, clamp_max);

    // Cast the reference output to F16
    auto ref_dst = cast<Float16, float>(ref_dst_float.data(), ref_dst_float.size() * 8 / size_in_bits<float>);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon(M, K, mr, kr, sr);
    std::vector<uint8_t> imp_packed_lhs(imp_packed_lhs_size);

    auto lhs_stride = K * sizeof(uint16_t);
    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_f16_neon(
        rect.height() /* m */, K, mr, kr, sr, 0, ref_lhs_f16.data() + lhs_offset, lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    const auto ref_rhs_qsi4_padded = pad_row<Int4>(
        ref_rhs_qsi4.data(), N, K, K, round_up_multiple(K, 2), round_up_division(N * round_up_multiple(K, 2), 2));

    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(N, K, nr, kr, sr);
    std::vector<uint8_t> imp_packed_rhs(imp_packed_rhs_size);
    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(rhs_start_row, K, nr, kr, sr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    // Runs the RHS packing micro-kernel.
    kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 0;

    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
        1, N, K, nr, kr, sr, ref_rhs_qsi4_padded.data(),
        has_bias ? reinterpret_cast<const float*>(ref_biases.data()) : nullptr,
        reinterpret_cast<const float*>(ref_rhs_scales.data()), imp_packed_rhs.data(), 0, &params);

    const auto dst_stride_row = N * sizeof(uint16_t);
    const auto dst_stride_col = sizeof(uint16_t);
    const auto dst_offset =
        ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride_row);
    const auto ref_dst_offset = rect.start_row() * dst_stride_row + rect.start_col() * dst_stride_col;
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    std::vector<uint8_t> imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        rect.height(), rect.width(), K, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, imp_dst.data() + dst_offset, dst_stride_row, dst_stride_col,
        clamp_min, clamp_max);

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    DataFormat dst_format = DataFormat(DataType::FP16);
    const auto success = compare(imp_dst.data(), ref_dst.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);
}
INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f16_qai8dxp_qsi4cxp,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f16_qai8dxp_qsi4cxp.size()),
        testing::Values(
            MatMulShape{1, 2, 32},     //
            MatMulShape{1, 3, 32},     //
            MatMulShape{1, 4, 32},     //
            MatMulShape{1, 5, 31},     //
            MatMulShape{3, 3, 32},     //
            MatMulShape{4, 4, 32},     //
            MatMulShape{5, 5, 31},     //
            MatMulShape{16, 32, 64},   //
            MatMulShape{16, 32, 36},   //
            MatMulShape{15, 35, 65},   //
            MatMulShape{8, 32, 64},    //
            MatMulShape{15, 31, 45},   //
            MatMulShape{1, 35, 65},    //
            MatMulShape{1, 128, 32},   //
            MatMulShape{64, 128, 32},  //
            MatMulShape{77, 99, 64}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),         // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),      // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),      // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8),     // Somewhere Middle
            MatrixPortion(0.75, 0.75, 1, 1),   // Bottom-right corner.
            MatrixPortion(0.75, 0, 1, 1),      // Partial rows
            MatrixPortion(0.4, 0.5, 0.6, 0.8)  // Somewhere Middle
            ),
        testing::Bool()),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f16_qai8dxp_qsi4cxp.at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<2>(info.param);
        const auto has_bias = std::get<3>(info.param);

        std::stringstream sstream;
        sstream << name << "__M_" << shape.m << "__N_" << shape.n << "__K_" << shape.k   //
                << "__PortionStartRow_" << static_cast<int>(portion.start_row() * 1000)  //
                << "__PortionStartCol_" << static_cast<int>(portion.start_col() * 1000)  //
                << "__PortionHeight_" << static_cast<int>(portion.height() * 1000)       //
                << "__PortionWidth_" << static_cast<int>(portion.width() * 1000)         //
                << (has_bias ? "__Bias" : "");
        return sstream.str();
    });

}  // namespace kai::test
