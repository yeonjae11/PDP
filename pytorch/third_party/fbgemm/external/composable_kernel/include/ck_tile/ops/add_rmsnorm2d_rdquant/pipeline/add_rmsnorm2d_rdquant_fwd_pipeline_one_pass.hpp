// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = AddRmsnorm2dRdquantFwdPipelineDefaultPolicy>
struct AddRmsnorm2dRdquantFwdPipelineOnePass
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using ADataType       = ck_tile::remove_cvref_t<typename Problem::ADataType>;
    using BDataType       = ck_tile::remove_cvref_t<typename Problem::BDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using YScaleDataType  = ck_tile::remove_cvref_t<typename Problem::YScaleDataType>;
    using QYDataType      = ck_tile::remove_cvref_t<typename Problem::QYDataType>;

    static constexpr bool kHasGamma = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kSaveX    = Problem::kSaveX;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM   = false; // TODO - BlockAddRmsnorm2dRdquantFwdProblem::kPadM
    static constexpr bool kPadN   = Problem::kPadN;
    static constexpr bool UseMax3 = true; // TODO - Move to trait

    static constexpr const char* name = []() {
        if constexpr(kNeedCrossWarpSync)
            return "bpr_op"; // block per row
        else
            return "wpr_op"; // warp per row
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename AWindow,
              typename BWindow,
              typename GammaWindow,
              typename XWindow,
              typename YScaleWindow,
              typename QYWindow>
    CK_TILE_DEVICE auto operator()(const AWindow& a_window_,
                                   const BWindow& b_window_,
                                   const GammaWindow& gamma_window_,
                                   XWindow& x_window,
                                   YScaleWindow& yscale_window,
                                   QYWindow& qy_window,
                                   ComputeDataType epsilon,
                                   ck_tile::index_t row_size,
                                   void* smem) const
    {
        const auto a_window =
            make_tile_window(a_window_, Policy::template MakeABXBlockTileDistribution<Problem>());
        const auto b_window =
            make_tile_window(b_window_, Policy::template MakeABXBlockTileDistribution<Problem>());
        const auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBlockTileDistribution<Problem>());

        auto reduce_square_sum_func = ReduceOp::SquareAdd{};
        auto reduce_sum_func        = ReduceOp::Add{};
        auto reduce_absmax_func     = ReduceOp::AbsMax{};
        auto reduce_absmax3_func    = [](auto acc_, auto v_0_, auto v_1_) {
            float rtn;
            asm volatile("v_max3_f32 %0, %1, abs(%2), abs(%3)"
                         : "=v"(rtn)
                         : "v"(acc_), "v"(v_0_), "v"(v_1_));
            return rtn;
        };
        auto reduce_max_func     = ReduceOp::Max{};
        auto block_reduce2d      = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        const auto a     = load_tile(a_window);
        const auto b     = load_tile(b_window);
        const auto gamma = load_tile(gamma_window);

        auto x = tile_elementwise_in(
            [&](const auto& a_, const auto& b_) {
                return type_convert<ComputeDataType>(a_) + type_convert<ComputeDataType>(b_);
            },
            a,
            b);

        if constexpr(kSaveX)
            store_tile(x_window, cast_tile<XDataType>(x));

        // compute mean square, each-thread->cross-lane->cross-warp
        auto square_sum = block_reduce2d(
            x, reduce_square_sum_func.GetIdentityValue<ComputeDataType>(), reduce_square_sum_func);
        block_reduce2d_sync(square_sum, reduce_sum_func);
        block_reduce2d_cross_warp_sync(square_sum, smem, reduce_sum_func);

        auto inv_rms = tile_elementwise_in(
            [&](const auto& v_) {
                return type_convert<ComputeDataType>(1.0f) / (sqrt(v_ / row_size + epsilon));
            },
            square_sum);

        // rmsnorm computation
        auto y = make_static_distributed_tensor<ComputeDataType>(x.get_tile_distribution());
        sweep_tile(y, [&, inv_rms_ = inv_rms](auto idx) {
            constexpr auto i_idx = make_tuple(idx[number<0>{}]);
            constexpr auto j_idx = make_tuple(idx[number<1>{}]);

            const auto gamma_ = type_convert<ComputeDataType>(gamma[j_idx]);

            const auto x_ = type_convert<ComputeDataType>(x[idx]);
            auto y_       = x_ * inv_rms_[i_idx] * gamma_;

            y(idx) = type_convert<ComputeDataType>(y_);
        });

        // compute absmax, each-thread->cross-lane->cross-warp
        auto absmax = [&]() {
            constexpr auto x_size_per_row =
                x.get_tile_distribution().get_ys_to_d_descriptor().get_lengths().at(number<1>{});
            if constexpr(UseMax3 && std::is_same_v<ComputeDataType, float> &&
                         x_size_per_row % 2 == 0)
            {
                return block_reduce2d(y,
                                      reduce_absmax_func.GetIdentityValue<ComputeDataType>(),
                                      reduce_absmax3_func,
                                      sequence<1, 2>{});
            }
            else
            {
                return block_reduce2d(
                    y, reduce_absmax_func.GetIdentityValue<ComputeDataType>(), reduce_absmax_func);
            }
        }();
        block_reduce2d_sync(absmax, reduce_max_func);
        block_reduce2d_cross_warp_sync(absmax, smem, reduce_max_func);

        // ex: yscale = absmax / 127 if int8
        auto yscale = tile_elementwise_in(
            [&](const auto& v_) {
                return v_ / type_convert<ComputeDataType>(numeric<QYDataType>::max());
            },
            absmax);
        store_tile(yscale_window, cast_tile<YScaleDataType>(yscale));

        // quantize y to qy
        auto qy = make_static_distributed_tensor<QYDataType>(y.get_tile_distribution());
        sweep_tile(qy, [&, yscale_ = yscale](auto idx) {
            constexpr auto i_idx = make_tuple(idx[number<0>{}]);
            auto qy_             = y[idx] / yscale_[i_idx];
            qy(idx)              = saturates<QYDataType>{}(qy_);
        });
        store_tile(qy_window, qy);
    }
};
} // namespace ck_tile
