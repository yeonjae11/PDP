// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <type_traits>

namespace ck_tile {
namespace element_wise {

#if 0
struct PassThroughPack2
{
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const;

    CK_TILE_HOST_DEVICE constexpr void operator()(ck_tile::half2_t& y, const ck_tile::f8x2_t& x) const
    {
        auto t = type_convert<float2_t>(x);
        y      = type_convert<half2_t>(t);
    }
    constexpr const static bool is_pack2_invocable = true;
};
#endif

struct PassThrough
{
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<double, double>(double& y, const double& x) const
    {
        y = x;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, double>(float& y, const double& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<double, float>(double& y, const float& x) const
    {
        y = type_convert<double>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        y = x;
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::fp16_t, ck_tile::fp16_t>(ck_tile::fp16_t& y, const ck_tile::fp16_t& x) const
    {
        y = x;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::fp16_t, float>(ck_tile::fp16_t& y,
                                                                const float& x) const
    {
        y = type_convert<ck_tile::fp16_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::bf16_t, ck_tile::bf16_t>(ck_tile::bf16_t& y, const ck_tile::bf16_t& x) const
    {
        y = x;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<int32_t, int32_t>(int32_t& y, const int32_t& x) const
    {
        y = x;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::bf16_t, float>(ck_tile::bf16_t& y,
                                                                const float& x) const
    {
        y = type_convert<ck_tile::bf16_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, ck_tile::bf16_t>(float& y,
                                                                const ck_tile::bf16_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::bf16_t, ck_tile::fp16_t>(ck_tile::bf16_t& y, const ck_tile::fp16_t& x) const
    {
        y = type_convert<ck_tile::bf16_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, ck_tile::fp16_t>(float& y,
                                                                const ck_tile::fp16_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<int8_t, int8_t>(int8_t& y, const int8_t& x) const
    {
        y = x;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::fp16_t, int8_t>(ck_tile::fp16_t& y,
                                                                 const int8_t& x) const
    {
        y = type_convert<ck_tile::fp16_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::bf16_t, int8_t>(ck_tile::bf16_t& y,
                                                                 const int8_t& x) const
    {
        y = type_convert<ck_tile::bf16_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<uint8_t, uint8_t>(uint8_t& y, const uint8_t& x) const
    {
        y = x;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<int8_t, int32_t>(int8_t& y, const int32_t& x) const
    {
        y = type_convert<int8_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<int32_t, int8_t>(int32_t& y, const int8_t& x) const
    {
        y = type_convert<int32_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<int8_t, float>(int8_t& y, const float& x) const
    {
        y = type_convert<int8_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, int8_t>(float& y, const int8_t& x) const
    {
        y = type_convert<float>(x);
    }

#ifdef CK_TILE_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    CK_TILE_HOST_DEVICE void operator()<int4_t, int4_t>(int4_t& y, const int4_t& x) const
    {
        y = x;
    }
    template <>
    CK_TILE_HOST_DEVICE void operator()<int4_t, int>(int4_t& y, const int& x) const
    {
        y = type_convert<int4_t>(x);
    }
#endif

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::fp8_t, ck_tile::fp8_t>(ck_tile::fp8_t& y, const ck_tile::fp8_t& x) const
    {
        y = x;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, ck_tile::fp8_t>(float& y,
                                                               const ck_tile::fp8_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::fp8_t, float>(ck_tile::fp8_t& y,
                                                               const float& x) const
    {
        y = type_convert<ck_tile::fp8_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::fp16_t, ck_tile::fp8_t>(ck_tile::fp16_t& y, const ck_tile::fp8_t& x) const
    {
        y = type_convert<ck_tile::fp16_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::fp8_t, ck_tile::fp16_t>(ck_tile::fp8_t& y, const ck_tile::fp16_t& x) const
    {
        y = type_convert<ck_tile::fp8_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::bf8_t, ck_tile::bf8_t>(ck_tile::bf8_t& y, const ck_tile::bf8_t& x) const
    {
        y = x;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, ck_tile::bf8_t>(float& y,
                                                               const ck_tile::bf8_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::bf8_t, float>(ck_tile::bf8_t& y,
                                                               const float& x) const
    {
        y = type_convert<ck_tile::bf8_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::fp16_t, ck_tile::bf8_t>(ck_tile::fp16_t& y, const ck_tile::bf8_t& x) const
    {
        y = type_convert<ck_tile::fp16_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::bf8_t, ck_tile::fp16_t>(ck_tile::bf8_t& y, const ck_tile::fp16_t& x) const
    {
        y = ck_tile::type_convert<ck_tile::bf8_t>(x);
    }
};

#if 0
struct UnaryConvert
{
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
    }
};

struct ConvertBF16RTN
{
    // convert to bf16 using round to nearest (rtn)
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(std::is_same_v<Y, ck_tile::bf16_t>, "Data type is not supported by this operation!");

        // check X datatype
        static_assert(std::is_same_v<X, float> || std::is_same_v<X, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        y = bf16_convert_rtn<Y>(x);
    }
};

struct ConvertF8SR
{
    // convert to fp8 using stochastic rounding (SR)
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(std::is_same_v<Y, ck_tile::fp8_t> || std::is_same_v<Y, ck_tile::bf8_t>,
                      "Data type is not supported by this operation!");

        // check X datatype
        static_assert(std::is_same_v<X, float> || std::is_same_v<X, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        y = f8_convert_sr<Y>(x);
    }
};

struct ConvertF8RNE
{
    // convert to fp8 using rounding to nearest even
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(std::is_same_v<Y, ck_tile::fp8_t> || std::is_same_v<Y, ck_tile::bf8_t>,
                      "Data type is not supported by this operation!");

        // check X datatype
        static_assert(std::is_same_v<X, float> || std::is_same_v<X, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        y = f8_convert_rne<Y>(x);
    }
};
#endif

struct Scale
{
    CK_TILE_HOST_DEVICE Scale(float scale = 1.f) : scale_(scale) {}

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        y = ck_tile::type_convert<Y>(ck_tile::type_convert<float>(x) * scale_);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::fp16_t, ck_tile::fp16_t>(ck_tile::fp16_t& y, const ck_tile::fp16_t& x) const
    {
        y = ck_tile::type_convert<ck_tile::fp16_t>(scale_) * x;
    };

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::bf16_t, ck_tile::bf16_t>(ck_tile::bf16_t& y, const ck_tile::bf16_t& x) const
    {
        const float x_tmp = ck_tile::type_convert<float>(x);
        const float y_tmp = scale_ * x_tmp;
        y                 = ck_tile::type_convert<ck_tile::bf16_t>(y_tmp);
    };

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        y = scale_ * x;
    };

    template <>
    CK_TILE_HOST_DEVICE void operator()<double, double>(double& y, const double& x) const
    {
        y = scale_ * x;
    };

    template <>
    CK_TILE_HOST_DEVICE void operator()<int8_t, int8_t>(int8_t& y, const int8_t& x) const
    {
        y = ck_tile::type_convert<int8_t>(scale_ * ck_tile::type_convert<float>(x));
    };

    float scale_;
};

struct ScaleAndResetNaNToMinusInfinity
{
    CK_TILE_HOST_DEVICE ScaleAndResetNaNToMinusInfinity(float scale) : scale_(scale) {}

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        y = ck_tile::isnan(x) ? -numeric<float>::infinity() : scale_ * x;
    };

    float scale_;
};

struct UnaryDivide
{
    CK_TILE_HOST_DEVICE UnaryDivide(const int32_t divider = 1) : divider_(divider) {}

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = x / type_convert<T>(divider_);
    };

    int32_t divider_ = 1;
};

struct UnarySquare
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, ck_tile::fp16_t> ||
                          std::is_same_v<T, double> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>
#ifdef CK_TILE_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                          || std::is_same_v<T, int4_t>
#endif
                      ,
                      "Data type is not supported by this operation!");
        y = x * x;
    };
};

struct UnaryAbs
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::abs(x);
    };
};

struct UnarySqrt
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Data type is not supported by this operation!");

        y = ck_tile::sqrt(x);
    };
};

struct Relu
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        y = x > 0 ? x : 0;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::bf16_t& y, const ck_tile::bf16_t& x) const
    {
        float x_f32 = ck_tile::type_convert<float>(x);
        float y_f32 = x_f32 > 0 ? x_f32 : 0;
        y           = ck_tile::type_convert<ck_tile::bf16_t>(y_f32);
    }
};

// Fast GeLU
// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// host code use higher accuracy "exp" and "div"
// gpu code use lower accuracy "_ocml_exp_f32" and "rcp" function
struct FastGelu
{
    template <typename Y, typename X>
    CK_TILE_HOST void operator()(Y& y, const X& x) const;

    template <typename Y, typename X>
    CK_TILE_DEVICE void operator()(Y& y, const X& x) const;

    template <>
    CK_TILE_HOST void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = -2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = exp(u);
        y               = x / (1.f + emu);
    }

    // device code, use lower precision "__ocml_exp_f32" and "rcp"
    template <>
    CK_TILE_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = __ocml_exp_f32(u);

        y = x * ck_tile::rcp(1.f + emu);
    }

    template <>
    CK_TILE_HOST void operator()<ck_tile::fp16_t, ck_tile::fp16_t>(ck_tile::fp16_t& y,
                                                                   const ck_tile::fp16_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<ck_tile::fp16_t>(y_f);
    }

    template <>
    CK_TILE_DEVICE void operator()<ck_tile::fp16_t, ck_tile::fp16_t>(ck_tile::fp16_t& y,
                                                                     const ck_tile::fp16_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<ck_tile::fp16_t>(y_f);
    }

    template <>
    CK_TILE_HOST void operator()<ck_tile::fp16_t, float>(ck_tile::fp16_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<ck_tile::fp16_t>(y_f);
    }

    template <>
    CK_TILE_DEVICE void operator()<ck_tile::fp16_t, float>(ck_tile::fp16_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<ck_tile::fp16_t>(y_f);
    }

    template <>
    CK_TILE_HOST void operator()<ck_tile::bf16_t, float>(ck_tile::bf16_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<ck_tile::bf16_t>(y_f);
    }

    template <>
    CK_TILE_DEVICE void operator()<ck_tile::bf16_t, float>(ck_tile::bf16_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<ck_tile::bf16_t>(y_f);
    }

    template <>
    CK_TILE_DEVICE void operator()<ck_tile::bf16_t, ck_tile::bf16_t>(ck_tile::bf16_t& y,
                                                                     const ck_tile::bf16_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<ck_tile::bf16_t>(y_f);
    }

    template <>
    CK_TILE_HOST void operator()<ck_tile::bf16_t, ck_tile::bf16_t>(ck_tile::bf16_t& y,
                                                                   const ck_tile::bf16_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<ck_tile::bf16_t>(y_f);
    }
};

struct FastGeluAsm
{
    template <typename Y, typename X>
    CK_TILE_HOST void operator()(Y& y, const X& x) const;

    template <typename Y, typename X>
    CK_TILE_DEVICE void operator()(Y& y, const X& x) const;

    template <>
    CK_TILE_HOST void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = -2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = exp(u);
        y               = x / (1.f + emu);
    }

    // device code, use lower precision "__ocml_exp_f32" and "rcp"
    template <>
    CK_TILE_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        const uint32_t c1     = 0xbd92220c; // -2.0 * 0.035677f;
        const float c2        = -2.0 * 0.797885f;
        const uint32_t log2e_ = 0x3fb8aa3b; // log2e_v<float>;
        float tmp;

        asm volatile("v_mul_f32 %[v_tmp], %[v_x], %[v_x]        ; x*x\n"
                     "v_fma_f32 %[v_tmp], %[v_tmp], %[s_c1], %[v_c2]  ; c1*x*x+c2\n"
                     "v_mul_f32 %[v_tmp], %[v_tmp], %[v_x]      ; x*(c1*x*x+c2)\n"
                     "v_mul_f32 %[v_tmp], %[v_tmp], %[s_log2e]  ; log2e*x*(c1*x*x+c2)\n"
                     "v_exp_f32 %[v_tmp], %[v_tmp]              ; emu = exp2(log2e*x*(c1*x*x+c2))\n"
                     "s_nop 0                                   ; hazard for exp\n"
                     "v_add_f32 %[v_tmp], %[v_tmp], 1.0         ; emu+1.0f\n"
                     "v_rcp_f32 %[v_tmp], %[v_tmp]              ; 1/(emu+1.0f)\n"
                     "s_nop 0                                   ; hazard for rcp \n"
                     "v_mul_f32 %[v_y], %[v_tmp], %[v_x]        ; x * 1/(emu+1f)\n"
                     : [v_y] "=v"(y), [v_tmp] "+v"(tmp)
                     : [v_x] "v"(x), [s_c1] "s"(c1), [v_c2] "v"(c2), [s_log2e] "s"(log2e_)
                     :);
    }

    template <>
    CK_TILE_HOST void operator()<fp32x2_t, fp32x2_t>(fp32x2_t& y, const fp32x2_t& x) const
    {
        const float c1   = -2.0 * 0.035677f;
        const float c2   = -2.0 * 0.797885f;
        const float u0   = x.x * (c1 * x.x * x.x + c2);
        const float emu0 = exp(u0);
        y.x              = x.x / (1.f + emu0);
        const float u1   = x.y * (c1 * x.y * x.y + c2);
        const float emu1 = exp(u1);
        y.y              = x.y / (1.f + emu1);
    }

    // this is packed verion to remove data hazard for trans
    template <>
    CK_TILE_DEVICE void operator()<fp32x2_t, fp32x2_t>(fp32x2_t& y, const fp32x2_t& x) const
    {
        const uint32_t c1     = 0xbd92220c; // -2.0 * 0.035677f;
        float c2              = -2.0 * 0.797885f;
        const uint32_t log2e_ = 0x3fb8aa3b; // log2e_v<float>;
        float tmp0, tmp1;
        float y0 = x.x, y1 = x.y;

        asm volatile(
            "v_mul_f32 %[v_tmp0], %[v_y0], %[v_y0]        ; x*x\n"
            "v_mul_f32 %[v_tmp1], %[v_y1], %[v_y1]        ; x*x\n"
            "v_fma_f32 %[v_tmp0], %[v_tmp0], %[s_c1], %[v_c2]  ; c1*x*x+c2\n"
            "v_fma_f32 %[v_tmp1], %[v_tmp1], %[s_c1], %[v_c2]  ; c1*x*x+c2\n"
            "v_mul_f32 %[v_tmp0], %[v_tmp0], %[v_y0]      ; x*(c1*x*x+c2)\n"
            "v_mul_f32 %[v_tmp1], %[v_tmp1], %[v_y1]      ; x*(c1*x*x+c2)\n"
            "v_mul_f32 %[v_tmp0], %[v_tmp0], %[s_log2e]  ; log2e*x*(c1*x*x+c2)\n"
            "v_mul_f32 %[v_tmp1], %[v_tmp1], %[s_log2e]  ; log2e*x*(c1*x*x+c2)\n"
            "v_exp_f32 %[v_tmp0], %[v_tmp0]              ; emu = exp2(log2e*x*(c1*x*x+c2))\n"
            "v_exp_f32 %[v_tmp1], %[v_tmp1]              ; emu = exp2(log2e*x*(c1*x*x+c2))\n"
            "v_add_f32 %[v_tmp0], %[v_tmp0], 1.0         ; emu+1.0f\n"
            "v_add_f32 %[v_tmp1], %[v_tmp1], 1.0         ; emu+1.0f\n"
            "v_rcp_f32 %[v_tmp0], %[v_tmp0]              ; 1/(emu+1.0f)\n"
            "v_rcp_f32 %[v_tmp1], %[v_tmp1]              ; 1/(emu+1.0f)\n"
            "v_mul_f32 %[v_y0], %[v_tmp0], %[v_y0]        ; x * 1/(emu+1f)\n"
            "v_mul_f32 %[v_y1], %[v_tmp1], %[v_y1]        ; x * 1/(emu+1f)\n"
            : [v_y0] "+v"(y0),
              [v_y1] "+v"(y1),
              [v_c2] "+v"(c2),
              // NOTE! it is totally possible that c2/y0/y1 share same register, they are all local
              // tmp variables we need to expicitly hint compiler they may read+write, to allow
              // allocate different register , the side effect is c2=** may issue for every such
              // inline asm block
              [v_tmp0] "+v"(tmp0),
              [v_tmp1] "+v"(tmp1)
            : [s_c1] "s"(c1), [s_log2e] "s"(log2e_)
            :);
        y.x = y0;
        y.y = y1;
    }
};

// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+erf(x/sqrt(2)))
struct Gelu
{
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        y = 0.5f * x * (1.f + erf(float(0.70710678118f * x)));
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::fp16_t, ck_tile::fp16_t>(ck_tile::fp16_t& y, const ck_tile::fp16_t& x) const
    {
        y = ck_tile::fp16_t(0.5) * x *
            (ck_tile::fp16_t(1) + ck_tile::fp16_t(erf(float(0.70710678118f * x))));
    }
};

struct Sigmoid
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");
        constexpr T one = type_convert<T>(1);
        y               = one / (one + ck_tile::exp(-x));
    };
};

struct Silu
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");
        constexpr T one = type_convert<T>(1);
        y               = x * (one / (one + ck_tile::exp(-x)));
    };
};

struct TanH
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::tanh(x);
    };
};

struct ACos
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::acos(x);
    };
};

struct Neg
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::neg(x);
    };
};

struct ATan
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::atan(x);
    };
};

struct Sin
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::sin(x);
    };
};

struct ASinH
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::asinh(x);
    };
};

struct Cos
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::cos(x);
    };
};

struct ACosH
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::acosh(x);
    };
};

struct Tan
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::tan(x);
    };
};

struct ATanH
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::atanh(x);
    };
};

struct SinH
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::sinh(x);
    };
};

struct Ceil
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::ceil(x);
    };
};

struct Exp
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::exp(x);
    };
};

struct CosH
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::cosh(x);
    };
};

struct Floor
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::floor(x);
    };
};

struct Log
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::log(x);
    };
};

struct ASin
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::asin(x);
    };
};

struct Rcp
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::rcp(x);
    };
};

struct Swish
{
    Swish(float beta = 1.0f) : beta_(beta) {}

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        static_assert(std::is_same_v<X, float> || std::is_same_v<X, double> ||
                          std::is_same_v<X, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        static_assert(std::is_same_v<Y, float> || std::is_same_v<Y, double> ||
                          std::is_same_v<Y, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<Y>(x / (1.f + ck_tile::exp(bx)));
    };

    const float beta_;
};

struct SoftRelu
{
    SoftRelu(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha  = type_convert<T>(alpha_);
        constexpr T one = type_convert<T>(1);
        y               = ck_tile::log(one + ck_tile::exp(x * casted_alpha)) / casted_alpha;
    }
    const float alpha_;
};

struct Power
{
    Power(float alpha = 0.f, float beta = 1.f, float gamma = 2.f)
        : alpha_(alpha), beta_(beta), gamma_(gamma){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha     = type_convert<T>(alpha_);
        T casted_beta      = type_convert<T>(beta_);
        T casted_gamma     = type_convert<T>(gamma_);
        T shifted_scaled_x = casted_alpha + casted_beta * x;
        y                  = ck_tile::pow(shifted_scaled_x, casted_gamma);
    }
    const float alpha_;
    const float beta_;
    const float gamma_;
};

struct ClippedRelu
{
    ClippedRelu(float alpha = 0.f, float beta = 1.f) : alpha_(alpha), beta_(beta){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        T casted_beta  = type_convert<T>(beta_);
        y              = ck_tile::min(casted_beta, ck_tile::max(casted_alpha, x));
    }
    const float alpha_;
    const float beta_;
};

struct LeakyRelu
{
    LeakyRelu(float alpha = 0.01f) : alpha_(alpha){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        y              = x >= 0 ? x : x * casted_alpha;
    }
    const float alpha_;
};

struct Elu
{
    Elu(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        y              = x > 0 ? x : casted_alpha * ck_tile::expm1(x);
    }
    const float alpha_;
};

struct Logistic
{
    Logistic(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha  = type_convert<T>(alpha_);
        constexpr T one = type_convert<T>(1);
        y               = casted_alpha / (one + ck_tile::exp(-x) * casted_alpha);
    }
    const float alpha_;
};

struct ConvInvscale
{
    CK_TILE_HOST_DEVICE
    ConvInvscale(float scale_in = 1.f, float scale_wei = 1.f, float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::fp8_t, float>(ck_tile::fp8_t& e,
                                                               const float& c) const
    {
        e = type_convert<ck_tile::fp8_t>(c / scale_in_ / scale_wei_ / scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

struct ConvScale
{
    CK_TILE_HOST_DEVICE
    ConvScale(float scale_in = 1.f, float scale_wei = 1.f, float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::fp8_t, float>(ck_tile::fp8_t& e,
                                                               const float& c) const
    {
        e = type_convert<ck_tile::fp8_t>(c * scale_in_ * scale_wei_ * scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

struct ConvScaleRelu
{
    CK_TILE_HOST_DEVICE
    ConvScaleRelu(float scale_in = 1.f, float scale_wei = 1.f, float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::fp8_t, float>(ck_tile::fp8_t& e,
                                                               const float& c) const
    {
        float x;
        Relu{}.template operator()<float>(x, c * scale_in_ * scale_wei_);
        e = type_convert<ck_tile::fp8_t>(x * scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

template <typename DstType, typename SrcType>
struct Cast
{
    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(DstType& y, const SrcType& x) const
    {
        y = ck_tile::type_convert<DstType>(x);
    };
};

// support fastconvert of int8 to fp16
#if 0
template <typename InputDataType, typename OutputDataType, index_t RegPackNumber>
struct FastNumericArrayConverter
{
};

template <>
struct FastNumericArrayConverter<uint8_t, ck_tile::fp16_t, 4>
{
    using InputArray  = vector_type<uint8_t, 4>;
    using OutputArray = vector_type<ck_tile::fp16_t, 4>;

    CK_TILE_DEVICE static OutputArray convert(InputArray const& Input)
    {
        OutputArray Output;

        uint32_t* half_2       = reinterpret_cast<uint32_t*>(&Output);
        uint32_t const uint8_4 = reinterpret_cast<uint32_t const&>(Input);

        static constexpr uint32_t byte_selector_01 = 0x05010500;
        static constexpr uint32_t byte_selector_23 = 0x05030502;
        static constexpr uint32_t fp16_adder       = 0x64646464;
        half_2[0] = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_01);
        half_2[1] = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_23);

        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
        asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                     : "=v"(half_2[0])
                     : "v"(half_2[0]), "s"(I8s_TO_F16s_MAGIC_NUM));
        asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                     : "=v"(half_2[1])
                     : "v"(half_2[1]), "s"(I8s_TO_F16s_MAGIC_NUM));

        return Output;
    }

    CK_TILE_DEVICE OutputArray operator()(InputArray const& Input) { return convert(Input); }
};

template <index_t N>
struct FastNumericArrayConverter<uint8_t, ck_tile::fp16_t, N>
{
    static constexpr int VEC_WIDTH = 4;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

    using InputArray  = vector_type<uint8_t, N>;
    using OutputArray = vector_type<ck_tile::fp16_t, N>;

    CK_TILE_DEVICE static OutputArray convert(InputArray const& Input)
    {
        FastNumericArrayConverter<uint8_t, ck_tile::fp16_t, 4> converter;

        OutputArray Output;

        using Vec_InputArray  = vector_type<uint8_t, 4>;
        using Vec_OutputArray = vector_type<ck_tile::fp16_t, 4>;

        Vec_OutputArray* half_4_ptr       = reinterpret_cast<Vec_OutputArray*>(&Output);
        Vec_InputArray const* uint8_4_ptr = reinterpret_cast<Vec_InputArray const*>(&Input);

        static_for<0, N / VEC_WIDTH, 1>{}(
            [&](auto i) { half_4_ptr[i] = converter(uint8_4_ptr[i]); });

        return Output;
    }

    CK_TILE_DEVICE OutputArray operator()(InputArray const& Input) { return convert(Input); }
};
#endif
} // namespace element_wise
} // namespace ck_tile
