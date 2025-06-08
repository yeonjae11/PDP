// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/random.hpp"
#include <stdint.h>
#include <type_traits>
#include "ck_tile/core/numeric/int8.hpp"

#pragma once

namespace ck_tile {

// Packed 2xint4
struct pk_int4_t
{
    using type = int8_t;
    type data;
    __host__ __device__ constexpr pk_int4_t() : data{type{}} {}
    __host__ __device__ constexpr pk_int4_t(type init) : data{init} {}
};

// limits
template <class T>
struct numeric;

template <>
struct numeric<pk_int4_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr pk_int4_t min()
    {
        constexpr uint8_t val = 0b10001000;
        return pk_int4_t(bit_cast<int8_t>(val));
    }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr pk_int4_t lowest()
    {
        constexpr uint8_t val = 0b10001000;
        return pk_int4_t(bit_cast<int8_t>(val));
    }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr pk_int4_t max()
    {
        constexpr uint8_t val = 0b01110111;
        return pk_int4_t(bit_cast<int8_t>(val));
    }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr pk_int4_t epsilon()
    {
        return 1; // not used
    }

    CK_TILE_HOST_DEVICE static constexpr pk_int4_t round_error()
    {
        return 1; // not used
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr pk_int4_t infinity()
    {
        return 1; // not used
    }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr pk_int4_t quiet_NaN()
    {
        return 1; // not used
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr pk_int4_t signaling_NaN()
    {
        return 1; // not used
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr pk_int4_t denorm_min()
    {
        return 1; // not used
    }

    CK_TILE_HOST_DEVICE static constexpr pk_int4_t zero() { return 0; }
};

CK_TILE_HOST_DEVICE fp32x2_t pk_int4_t_to_fp32x2_t(const pk_int4_t& x)
{
    uint8_t x_u8 = ck_tile::bit_cast<uint8_t>(x);

    float x_l = ((x_u8 & 0x0f) >> 0) - 8.f;
    float x_h = ((x_u8 & 0xf0) >> 4) - 8.f;

#ifdef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    fp32x2_t res = {x_h, x_l};
#elif
    fp32x2_t res = {x_l, x_h};
#endif
    return res;
}

CK_TILE_HOST_DEVICE fp16x2_t pk_int4_t_to_halfx2_t(const pk_int4_t& x)
{
    uint8_t x_u8 = ck_tile::bit_cast<uint8_t>(x);
#ifdef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    uint32_t i4s = ((x_u8 & 0x0f) << 16) | ((x_u8 & 0xf0) >> 4);
#elif
    uint32_t i4s = ((x_u8 & 0xf0) << 12) | (x_u8 & 0xf);
#endif
    const int EX  = 0x64006400;
    const int SUB = 0xE408E408; //-8

    int lo = i4s | EX;

    return pk_add_f16(bit_cast<fp16x2_t>(lo), bit_cast<fp16x2_t>(SUB));
}

CK_TILE_HOST_DEVICE bf16x2_t pk_int4_t_to_bfloat16x2_t(const pk_int4_t& x)
{
    uint8_t x_u8 = ck_tile::bit_cast<uint8_t>(x);

    float x_l = ((x_u8 & 0x0f) >> 0) - 8.f;
    float x_h = ((x_u8 & 0xf0) >> 4) - 8.f;

#ifdef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    bf16x2_t res = {type_convert<bf16_t>(x_h), type_convert<bf16_t>(x_l)};
#elif
    bf16x2_t res = {type_convert<bf16_t>(x_l), type_convert<bf16_t>(x_h)};
#endif
    return res;
}

} // namespace ck_tile
