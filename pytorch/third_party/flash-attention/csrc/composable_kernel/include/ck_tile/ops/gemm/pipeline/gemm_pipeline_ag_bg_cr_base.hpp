// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmPipelineAgBgCrImplBase
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    template <typename DstBlockTile, typename SrcTileWindow>
    CK_TILE_DEVICE void GlobalPrefetch(DstBlockTile& dst_block_tile,
                                       SrcTileWindow& dram_tile_window) const
    {
        load_tile(dst_block_tile, dram_tile_window);
        move_tile_window(dram_tile_window, {0, KPerBlock});
    }

    template <typename DstTileWindow, typename SrcBlockTile, typename ElementFunction>
    CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                     const SrcBlockTile& src_block_tile,
                                     const ElementFunction& element_func) const
    {
        const auto block_tile_tmp = tile_elementwise_in(element_func, src_block_tile);
        store_tile(lds_tile_window, block_tile_tmp);
    }

    CK_TILE_DEVICE auto GetABLdsTensorViews(void* p_smem) const
    {
        // A tile in LDS
        ADataType* p_a_lds              = static_cast<ADataType*>(p_smem);
        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        // TODO: LDS alignment should come from Policy!
        constexpr index_t a_lds_block_space_size_aligned =
            integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16) *
            16;

        // B tile in LDS
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        return make_tuple(std::move(a_lds_block), std::move(b_lds_block));
    }

    template <typename ADramBlockWindowTmp, typename ALdsTensorView>
    CK_TILE_DEVICE auto GetAWindows(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                    const ALdsTensorView& a_lds_block_view) const
    {
        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A LDS tile window for store
        auto a_copy_lds_window =
            make_tile_window(a_lds_block_view,
                             make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                             {0, 0},
                             a_copy_dram_window.get_tile_distribution());

        auto a_lds_gemm_window = make_tile_window(
            a_lds_block_view, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});

        return make_tuple(std::move(a_copy_dram_window),
                          std::move(a_copy_lds_window),
                          std::move(a_lds_gemm_window));
    }

    template <typename BDramBlockWindowTmp, typename BLdsTensorView>
    CK_TILE_DEVICE auto GetBWindows(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                    const BLdsTensorView& b_lds_block_view) const
    {
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window =
            make_tile_window(b_lds_block_view,
                             make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.get_tile_distribution());

        auto b_lds_gemm_window = make_tile_window(
            b_lds_block_view, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});

        return make_tuple(std::move(b_copy_dram_window),
                          std::move(b_copy_lds_window),
                          std::move(b_lds_gemm_window));
    }
};

} // namespace ck_tile
