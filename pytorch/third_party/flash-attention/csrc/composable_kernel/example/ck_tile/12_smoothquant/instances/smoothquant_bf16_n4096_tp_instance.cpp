
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "smoothquant_instance_common.hpp"

// clang-format off
//                                                   rm rn tm  tn   vn   pd    2p
template float smoothquant_<trait_<ck_tile::bf16_t,  1, 2, 1,  256, 8,  true, true>>(const S&, A);
template float smoothquant_<trait_<ck_tile::bf16_t,  1, 4, 1,  256, 4,  true, true>>(const S&, A);
template float smoothquant_<trait_<ck_tile::bf16_t,  1, 2, 1, 1024, 2,  true, true>>(const S&, A);
template float smoothquant_<trait_<ck_tile::bf16_t,  1, 4, 1, 1024, 1,  true, true>>(const S&, A);

// clang-format on
