// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// Y = X * XScale, QY = RowwiseDynamicQuant(Y) = SaturateCast(Y / YScale)
template <typename XDataType_,
          typename XScaleDataType_,
          typename ComputeDataType_,
          typename YScaleDataType_,
          typename QYDataType_,
          typename BlockShape_,
          bool kPadN_,
          bool kTwoPass_>
struct SmoothquantPipelineProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using XScaleDataType  = remove_cvref_t<XScaleDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YScaleDataType  = remove_cvref_t<YScaleDataType_>;
    using QYDataType      = remove_cvref_t<QYDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;

    static constexpr bool kNeedCrossLaneSync = BlockShape::ThreadPerWarp_N > 1;
    static constexpr bool kNeedCrossWarpSync = BlockShape::WarpPerBlock_N > 1;

    static constexpr bool kPadN    = kPadN_;
    static constexpr bool kTwoPass = kTwoPass_;
};

} // namespace ck_tile
