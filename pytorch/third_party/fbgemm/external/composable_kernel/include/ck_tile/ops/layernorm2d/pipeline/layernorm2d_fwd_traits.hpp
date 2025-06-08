// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

enum class Layernorm2dFusedAddEnum
{
    NO_ADD = 0,
    // fused add before layernorm and store result to global
    PRE_ADD_STORE = 1,
    // fused add before layernorm, but not store result
    PRE_ADD = 2,
};

// clang-format off
template<Layernorm2dFusedAddEnum> struct Layernorm2dFusedAddEnumName;
template<> struct Layernorm2dFusedAddEnumName<Layernorm2dFusedAddEnum::NO_ADD> { static constexpr const char * name = "no"; };
template<> struct Layernorm2dFusedAddEnumName<Layernorm2dFusedAddEnum::PRE_ADD_STORE> { static constexpr const char * name = "pras"; };
template<> struct Layernorm2dFusedAddEnumName<Layernorm2dFusedAddEnum::PRE_ADD> { static constexpr const char * name = "pra"; };
// clang-format on

enum class Layernorm2dFusedQuantEnum
{
    NO_SWEEP             = 0,
    SMOOTH_DYNAMIC_QUANT = 1, // smooth oulier + rowwise quant, need input x-scale and store y_scale
    DYNAMIC_QUANT        = 2, // rowwise quant, store out a y-scale
};

// clang-format off
template<Layernorm2dFusedQuantEnum> struct Layernorm2dFusedQuantEnumName;
template<> struct Layernorm2dFusedQuantEnumName<Layernorm2dFusedQuantEnum::NO_SWEEP> { static constexpr const char * name = "no"; };
template<> struct Layernorm2dFusedQuantEnumName<Layernorm2dFusedQuantEnum::DYNAMIC_QUANT> { static constexpr const char * name = "dqt"; };
template<> struct Layernorm2dFusedQuantEnumName<Layernorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT> { static constexpr const char * name = "smdqt"; };
// clang-format on

template <bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kFastFDiv_,
          bool kTwoPass_,
          Layernorm2dFusedAddEnum kFusedAdd_,
          Layernorm2dFusedQuantEnum kFusedQuant_>
struct Layernorm2dFwdTraits
{
    static constexpr bool kPadN                            = kPadN_;
    static constexpr bool kSaveMeanInvStd                  = kSaveMeanInvStd_;
    static constexpr bool kFastFDiv                        = kFastFDiv_;
    static constexpr bool kTwoPass                         = kTwoPass_;
    static constexpr Layernorm2dFusedAddEnum kFusedAdd     = kFusedAdd_;
    static constexpr Layernorm2dFusedQuantEnum kFusedQuant = kFusedQuant_;
};

} // namespace ck_tile
