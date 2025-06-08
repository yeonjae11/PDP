// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

struct GemmProblem
{
    CK_TILE_HOST GemmProblem() = default;
    CK_TILE_HOST GemmProblem(
        index_t M_, index_t N_, index_t K_, index_t stride_A_, index_t stride_B_, index_t stride_C_)
        : M(M_), N(N_), K(K_), stride_A(stride_A_), stride_B(stride_B_), stride_C(stride_C_)
    {
    }

    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
};

struct GemmHostArgs : public GemmProblem
{
    CK_TILE_HOST GemmHostArgs() = default;
    CK_TILE_HOST GemmHostArgs(const void* a_ptr_,
                              const void* b_ptr_,
                              void* c_ptr_,
                              index_t k_batch_,
                              index_t M_,
                              index_t N_,
                              index_t K_,
                              index_t stride_A_,
                              index_t stride_B_,
                              index_t stride_C_)
        : GemmProblem(M_, N_, K_, stride_A_, stride_B_, stride_C_),
          a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          c_ptr(c_ptr_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    index_t k_batch;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GemmKernel
{
    using TilePartitioner                    = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline                       = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline                   = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                            = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout                            = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout                            = remove_cvref_t<typename GemmPipeline::CLayout>;
    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();

    __host__ static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return TilePartitioner::GridSize(M, N, KBatch);
    }

    __host__ static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    struct GemmKernelArgs
    {
        const void* a_ptr;
        const void* b_ptr;
        void* c_ptr;
        index_t M;
        index_t N;
        index_t K;
        index_t stride_A;
        index_t stride_B;
        index_t stride_C;
        index_t KBatch;
    };

    CK_TILE_HOST static constexpr GemmKernelArgs MakeKernelArgs(const GemmHostArgs& hostArgs)
    {
        return GemmKernelArgs{hostArgs.a_ptr,
                              hostArgs.b_ptr,
                              hostArgs.c_ptr,
                              hostArgs.M,
                              hostArgs.N,
                              hostArgs.K,
                              hostArgs.stride_A,
                              hostArgs.stride_B,
                              hostArgs.stride_C,
                              hostArgs.k_batch};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const GemmKernelArgs& kargs,
                                     const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = kargs.KBatch * K1;
            const index_t KRead = (kargs.K + K_t - 1) / K_t * K1;

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = k_id * KRead;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = k_id * KRead * kargs.stride_A;
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = k_id * KRead * kargs.stride_B;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = k_id * KRead;
            }

            if(k_id < static_cast<uint32_t>(kargs.KBatch - 1))
            {
                splitted_k = KRead;
            }
            else
            {
                splitted_k = kargs.K - KRead * (kargs.KBatch - 1);
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_HOST static bool IsSupportedArgument(const GemmKernelArgs& kargs)
    {
        constexpr bool is_output_c_reg_transposed =
            EpiloguePipeline::IsOutputTransposed() != GemmPipeline::IsTransposeC();
        if constexpr(!((GemmPipeline::VectorSizeC % 2 == 0 &&
                        std::is_same_v<CLayout, tensor_layout::gemm::RowMajor> &&
                        is_output_c_reg_transposed) ||
                       !(std::is_same_v<CDataType, fp16_t> || std::is_same_v<CDataType, bf16_t>)))
        {
            if(kargs.KBatch != 1)
            {
                return false;
            }
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % TilePartitioner::kK != 0 && GemmPipeline::kPadK == false)
            {
                return false;
            }
            if(kargs.K % GemmPipeline::VectorSizeA != 0)
            {
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::kM != 0 && GemmPipeline::kPadM == false)
            {
                return false;
            }
            if(kargs.M % GemmPipeline::VectorSizeA != 0)
            {
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::kN != 0 && GemmPipeline::kPadN == false)
            {
                return false;
            }
            if(kargs.N % GemmPipeline::VectorSizeB != 0)
            {
                return false;
            }
        }
        else
        {
            if(kargs.K % TilePartitioner::kK != 0 && GemmPipeline::kPadK == false)
            {
                return false;
            }
            if(kargs.K % GemmPipeline::VectorSizeB != 0)
            {
                return false;
            }
        }

        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::kN != 0 && GemmPipeline::kPadN == false)
            {
                return false;
            }
            if(kargs.N % GemmPipeline::VectorSizeC != 0)
            {
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::kM != 0 && GemmPipeline::kPadM == false)
            {
                return false;
            }
            if(kargs.M % GemmPipeline::VectorSizeC != 0)
            {
                return false;
            }
        }
        return true;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto MakeGemmTensorViews(const ADataType* a_ptr,
                                                   const BDataType* b_ptr,
                                                   CDataType* c_ptr,
                                                   const GemmKernelArgs& kargs,
                                                   const SplitKBatchOffset& splitk_batch_offset)
    {
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::VectorSizeA>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                    make_tuple(1, kargs.stride_A),
                    number<1>{},
                    number<1>{});
            }
        }();

        const auto& b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_ptr,
                    make_tuple(kargs.N, splitk_batch_offset.splitted_k),
                    make_tuple(1, kargs.stride_B),
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_ptr,
                    make_tuple(kargs.N, splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_B, 1),
                    number<GemmPipeline::VectorSizeB>{},
                    number<1>{});
            }
        }();

        const auto& c_tensor_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<GemmPipeline::VectorSizeC>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();

        return make_tuple(a_tensor_view, b_tensor_view, c_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(
                    a_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
                    sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(
                    a_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
                    sequence<GemmPipeline::kPadM, false>{});
            }
        }();

        const auto& b_pad_view = [&]() {
            const auto& b_tensor_view = views.at(I1);
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return pad_tensor_view(
                    b_tensor_view,
                    make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
                    sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(
                    b_tensor_view,
                    make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
                    sequence<GemmPipeline::kPadN, false>{});
            }
        }();

        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view = views.at(I2);
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(
                    c_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                    sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(
                    c_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                    sequence<GemmPipeline::kPadM, false>{});
            }
        }();

        return make_tuple(a_pad_view, b_pad_view, c_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& a_pad_view     = views.at(I0);
        const auto& a_block_window = make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
            {i_m, 0});

        const auto& b_pad_view     = views.at(I1);
        const auto& b_block_window = make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
            {i_n, 0});

        const auto& c_pad_view = views.at(I2);
        auto c_block_window    = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            {i_m, i_n});

        return make_tuple(a_block_window, b_block_window, c_block_window);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param c_ptr output C pointer
     * @param kargs GEMM kernel arguments
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     * @tparam DstInMemOp Destination memory operation (default: set).
     */
    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static void RunGemm(const ADataType* a_ptr,
                                       const BDataType* b_ptr,
                                       CDataType* c_ptr,
                                       void* smem_ptr,
                                       const GemmKernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<DstInMemOp>(a_ptr, b_ptr, c_ptr, kargs, splitk_batch_offset);
        ;
        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(I0);
        const auto& b_block_window = gemm_tile_windows.at(I1);
        const auto& c_block_tile =
            GemmPipeline{}.template operator()(a_block_window, b_block_window, num_loop, smem_ptr);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I2);

        constexpr bool is_output_c_reg_transposed =
            EpiloguePipeline::IsOutputTransposed() != GemmPipeline::IsTransposeC();
        if constexpr((DstInMemOp == memory_operation_enum::set) || (sizeof(CDataType) > 2) ||
                     (GemmPipeline::VectorSizeC % 2 == 0 &&
                      std::is_same_v<CLayout, tensor_layout::gemm::RowMajor> &&
                      is_output_c_reg_transposed))
        {
            EpiloguePipeline{}
                .template operator()<decltype(c_block_window), decltype(c_block_tile), DstInMemOp>(
                    c_block_window, c_block_tile);
        }
    }

    CK_TILE_DEVICE void operator()(GemmKernelArgs kargs) const
    {
        const auto [i_m, i_n] = TilePartitioner{}();
        const SplitKBatchOffset splitk_batch_offset(kargs);
        // options
        const ADataType* a_ptr =
            static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_ptr =
            static_cast<const BDataType*>(kargs.b_ptr) + splitk_batch_offset.b_k_split_offset;
        CDataType* c_ptr = static_cast<CDataType*>(kargs.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        if(kargs.KBatch == 1)
        {
            RunGemm(a_ptr, b_ptr, c_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
        }
        else
        {
            RunGemm<memory_operation_enum::atomic_add>(
                a_ptr, b_ptr, c_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
        }
    }
};

} // namespace ck_tile
