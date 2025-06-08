// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/random_gen.hpp"
#include "ck/utility/type.hpp"

#ifdef CK_USE_FNUZ_FP8
#define CK_USE_FNUZ_FP8 1
#else
#define CK_USE_FNUZ_FP8 0
#endif

#ifdef CK_USE_OCP_FP8
#define CK_USE_OCP_FP8 1
#else
#define CK_USE_OCP_FP8 0
#endif

namespace {
// https://en.cppreference.com/w/cpp/types/conditional
template <bool B, class T, class F>
struct conditional
{
    using type = T;
};
template <class T, class F>
struct conditional<false, T, F>
{
    using type = F;
};
} // namespace

namespace ck {

using f8_fnuz_t  = _BitInt(8);
using bf8_fnuz_t = unsigned _BitInt(8);

#if(defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx1200__) || \
    defined(__gfx1201__)) &&                                                                     \
    __HIP_DEVICE_COMPILE__
#define CK_FP8_CVT_FAST_PATH 1
#else
#define CK_FP8_CVT_FAST_PATH 0
#endif

#if(defined(__gfx1200__) || defined(__gfx1201__)) && __HIP_DEVICE_COMPILE__
#define CK_OCP_FP8_CVT_FAST_PATH 1
#else
#define CK_OCP_FP8_CVT_FAST_PATH 0
#endif

typedef unsigned char fp8_storage_t;

/**
 * \brief Describes FP8 interpretation
 */
enum class ck_fp8_interpretation_t
{
    CK_E4M3_OCP  = 0, // OCP E4M3
    CK_E5M2_OCP  = 1, // OCP E5M2
    CK_E4M3_FNUZ = 2, // FP8
    CK_E5M2_FNUZ = 3, // BF8
};

/**
 * \brief Describes saturation behavior
 */
enum class ck_saturation_t
{
    CK_NOSAT     = 0, // No saturation - replace with NaN or Inf
    CK_SATFINITE = 1, // Saturate to finite
};

namespace fp8_impl {

typedef fp8_storage_t fp8x2_storage_t __attribute__((ext_vector_type(2)));
typedef float float2_t __attribute__((ext_vector_type(2)));

__host__ __device__ static inline constexpr bool fnuz_f8_is_nan(f8_fnuz_t a)
{
    return static_cast<unsigned char>(a) == 0x80;
}
__host__ __device__ static inline constexpr bool fnuz_bf8_is_nan(bf8_fnuz_t a)
{
    return static_cast<unsigned char>(a) == 0x80;
}

__host__ __device__ static inline constexpr bool ocp_f8_is_nan(fp8_storage_t a)
{
    return (a & 0x7f) == 0x7f;
}
__host__ __device__ static inline constexpr bool ocp_bf8_is_nan(fp8_storage_t a)
{
    return (a & 0x7f) > 0x7c;
}

// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_hip_f8_impl.h#L220
// This has been modified to handle double types as well
template <typename T, int wm, int we, bool is_fnuz, bool clip = false>
__host__ __device__ static inline T cast_from_f8(fp8_storage_t x)
{
    constexpr bool is_half   = __hip_internal::is_same<T, _Float16>::value;
    constexpr bool is_float  = __hip_internal::is_same<T, float>::value;
    constexpr bool is_double = __hip_internal::is_same<T, double>::value;
    static_assert(is_half || is_float || is_double, "only half, float and double are supported");

    constexpr int weo = is_half ? 5 : (is_float ? 8 : 11);
    constexpr int wmo = is_half ? 10 : (is_float ? 23 : 52);

    T fInf, fNegInf, fNaN, fNeg0, fmax, fmin;
    if constexpr(is_half)
    {
        const unsigned short int ihInf    = 0x7C00;
        const unsigned short int ihNegInf = 0xFC00;
        const unsigned short int ihNaN    = 0x7C01;
        const unsigned short int ihNeg0   = 0x8000;
        /* Max number in e5m2 57344*/
        const unsigned short int ifmax = 0x7B00;
        const unsigned short int ifmin = 0xFB00;

        fInf    = bit_cast<_Float16>(ihInf);
        fNegInf = bit_cast<_Float16>(ihNegInf);
        fNaN    = bit_cast<_Float16>(ihNaN);
        fNeg0   = bit_cast<_Float16>(ihNeg0);
        fmax    = bit_cast<_Float16>(ifmax);
        fmin    = bit_cast<_Float16>(ifmin);
    }
    else if constexpr(is_float)
    {
        const unsigned int ifInf    = 0x7F800000;
        const unsigned int ifNegInf = 0xFF800000;
        const unsigned int ifNaN    = 0x7F800001;
        const unsigned int ifNeg0   = 0x80000000;
        /* Max number in e5m2 57344*/
        const unsigned int ifmax = 0x47600000;
        const unsigned int ifmin = 0xC7600000;

        fInf    = bit_cast<float>(ifInf);
        fNegInf = bit_cast<float>(ifNegInf);
        fNaN    = bit_cast<float>(ifNaN);
        fNeg0   = bit_cast<float>(ifNeg0);
        fmax    = bit_cast<float>(ifmax);
        fmin    = bit_cast<float>(ifmin);
    }
    else if constexpr(is_double)
    {
        const unsigned long long ifInf    = 0x7FF0000000000000ull;
        const unsigned long long ifNegInf = 0xFFF0000000000000ull;
        const unsigned long long ifNaN    = 0x7FF0000000000001ull;
        const unsigned long long ifNeg0   = 0x8000000000000000ull;
        /* Max number in e5m2 57344*/
        const unsigned long long ifmax = 0x40EC000000000000ull;
        const unsigned long long ifmin = 0xC0EC000000000000ull;

        fInf    = bit_cast<double>(ifInf);
        fNegInf = bit_cast<double>(ifNegInf);
        fNaN    = bit_cast<double>(ifNaN);
        fNeg0   = bit_cast<double>(ifNeg0);
        fmax    = bit_cast<double>(ifmax);
        fmin    = bit_cast<double>(ifmin);
    }

    if(x == 0)
    {
        return 0;
    }

    unsigned long long sign     = x >> 7;
    unsigned long long mantissa = x & ((1 << wm) - 1);
    int exponent                = (x & 0x7F) >> wm;
    if constexpr(is_fnuz)
    {
        if(x == 0x80)
        {
            return fNaN;
        }
    }
    else
    {
        if(x == 0x80)
        {
            return fNeg0;
        }
        if constexpr(we == 4)
        { // e4m3
            if((x & 0x7F) == 0x7F)
            {
                return fNaN;
            }
        }
        else if((x & 0x7C) == 0x7C)
        { // e5m2
            if((x & 0x3) == 0)
            {
                if constexpr(clip)
                {
                    return sign ? fmin : fmax;
                }
                return sign ? fNegInf : fInf;
            }
            return fNaN;
        }
    }

    typename conditional<
        sizeof(T) == 2,
        unsigned short int,
        typename conditional<sizeof(T) == 4, unsigned int, unsigned long long>::type>::type retval;

    if constexpr(we == 5 && is_half && !is_fnuz)
    {
        retval = x << 8;
        return bit_cast<T>(retval);
    }

    const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (is_fnuz ? 1 : 0);

    // subnormal input
    if(exponent == 0)
    {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + __clz(mantissa) - (32 - wm);
#else
        int sh = 1 + __builtin_clz(mantissa) - (32 - wm);
#endif
        mantissa <<= sh;
        exponent += 1 - sh;
        mantissa &= ((1ull << wm) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= wmo - wm;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << wmo;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    if constexpr(sizeof(T) == 2)
        retval = (sign << 15) | (exponent << 10) | mantissa;
    else if constexpr(sizeof(T) == 4)
        retval = (sign << 31) | (exponent << 23) | mantissa;
    else
        retval = (sign << 63) | (static_cast<unsigned long long>(exponent) << 52) | mantissa;

    return bit_cast<T>(retval);
}

#if CK_FP8_CVT_FAST_PATH
template <ck_fp8_interpretation_t interpret>
static __device__ float cast_to_f32_from_f8(fp8_storage_t v)
{
    union
    {
        unsigned int i32val;
        unsigned char i8val[4];
    } val;
    val.i8val[0] = v;

    static_assert(interpret == ck_fp8_interpretation_t::CK_E4M3_FNUZ ||
                      interpret == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_FNUZ ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only FNUZ and OCP interpretations are supported");

    if constexpr((interpret == ck_fp8_interpretation_t::CK_E4M3_FNUZ) ||
                 (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP))
    {
        return __builtin_amdgcn_cvt_f32_fp8(val.i32val, 0);
    }
    else
    {
        return __builtin_amdgcn_cvt_f32_bf8(val.i32val, 0);
    }
}

template <ck_fp8_interpretation_t interpret>
static __device__ float2_t cast_to_f32x2_from_f8x2(fp8x2_storage_t v)
{
    const auto i16val = bit_cast<uint16_t>(v);

    static_assert(interpret == ck_fp8_interpretation_t::CK_E4M3_FNUZ ||
                      interpret == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_FNUZ ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only FNUZ and OCP interpretations are supported");

    if constexpr((interpret == ck_fp8_interpretation_t::CK_E4M3_FNUZ) ||
                 (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP))
    {
        return __builtin_amdgcn_cvt_pk_f32_fp8(i16val, false);
    }
    else
    {
        return __builtin_amdgcn_cvt_pk_f32_bf8(i16val, false);
    }
}

#endif

} // namespace fp8_impl

struct f8_ocp_t
{
    using data_type = fp8_storage_t;
    data_type data;

    static constexpr ck_saturation_t default_saturation = ck_saturation_t::CK_SATFINITE;
    static constexpr ck_fp8_interpretation_t default_interpret =
        ck_fp8_interpretation_t::CK_E4M3_OCP;

    static constexpr unsigned int we = 4; // exponent width
    static constexpr unsigned int wm = 3; // mantissa width

    __host__ __device__ constexpr bool operator==(const f8_ocp_t& other) const
    {
        return (data == other.data) && (fp8_impl::ocp_f8_is_nan(data) == false); // NaN != NaN
    }

#if CK_USE_OCP_FP8
    __host__ __device__ explicit operator float() const
#else
    __host__ explicit operator float() const
#endif
    {
#if CK_OCP_FP8_CVT_FAST_PATH
        return fp8_impl::cast_to_f32_from_f8<default_interpret>(this->data);
#else
        return fp8_impl::cast_from_f8<float, wm, we, false>(
            this->data); // XXX: clip==false must be consistent with operator _Float16
#endif
    }

#if CK_USE_OCP_FP8
    __host__ __device__ explicit operator _Float16() const
#else
    __host__ explicit operator _Float16() const
#endif
    {
#if CK_OCP_FP8_CVT_FAST_PATH
        return static_cast<_Float16>(fp8_impl::cast_to_f32_from_f8<default_interpret>(this->data));
#else
        return fp8_impl::cast_from_f8<_Float16, wm, we, false>(
            this->data); // XXX: clip==false must be consistent with operator float
#endif
    }
};

struct bf8_ocp_t
{
    using data_type = fp8_storage_t;
    data_type data;

    static constexpr ck_saturation_t default_saturation = ck_saturation_t::CK_SATFINITE;
    static constexpr ck_fp8_interpretation_t default_interpret =
        ck_fp8_interpretation_t::CK_E5M2_OCP;

    static constexpr unsigned int we = 5; // exponent width
    static constexpr unsigned int wm = 2; // mantissa width

    __host__ __device__ constexpr bool operator==(const bf8_ocp_t& other) const
    {
        return (data == other.data) && (fp8_impl::ocp_bf8_is_nan(data) == false); // NaN != NaN
    }

#if CK_USE_OCP_FP8
    __host__ __device__ explicit operator float() const

#else
    __host__ explicit operator float() const
#endif
    {
#if defined(__gfx1200__) || defined(__gfx1201__)
        return fp8_impl::cast_to_f32_from_f8<default_interpret>(this->data);
#else
        return fp8_impl::cast_from_f8<float, wm, we, false>(
            this->data); // XXX: clip==false must be consistent with operator _Float16
#endif
    }

#if CK_USE_OCP_FP8
    __host__ __device__ explicit operator _Float16() const
#else
    __host__ explicit operator _Float16() const
#endif
    {
#if defined(__gfx1200__) || defined(__gfx1201__)
        return static_cast<_Float16>(fp8_impl::cast_to_f32_from_f8<default_interpret>(this->data));
#else
        return fp8_impl::cast_from_f8<_Float16, wm, we, false>(
            this->data); // XXX: clip==false must be consistent with operator float
#endif
    }
};

template <typename T>
__host__ __device__ static inline constexpr bool fp8_is_nan(T);

template <>
__host__ __device__ inline constexpr bool fp8_is_nan(f8_ocp_t a)
{
    return fp8_impl::ocp_f8_is_nan(a.data);
}
template <>
__host__ __device__ inline constexpr bool fp8_is_nan(bf8_ocp_t a)
{
    return fp8_impl::ocp_bf8_is_nan(a.data);
}
template <>
__host__ __device__ inline constexpr bool fp8_is_nan(f8_fnuz_t a)
{
    return fp8_impl::fnuz_f8_is_nan(a);
}
template <>
__host__ __device__ inline constexpr bool fp8_is_nan(bf8_fnuz_t a)
{
    return fp8_impl::fnuz_bf8_is_nan(a);
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, bf8_ocp_t> || std::is_same_v<T, f8_ocp_t> ||
                               std::is_same_v<T, bf8_fnuz_t> || std::is_same_v<T, f8_fnuz_t>,
                           bool> = true>
__host__ __device__ static inline constexpr bool fp8_is_inf(T)
{
    return false;
}
template <>
__host__ __device__ inline constexpr bool fp8_is_inf(bf8_ocp_t a)
{
    return (a.data & 0x7f) == 0x7c;
}

namespace fp8_impl {

// Assertions to check for supported conversion types
#define __assert_ocp_support(interp)                                               \
    {                                                                              \
        if(interp != ck_fp8_interpretation_t::CK_E4M3_OCP &&                       \
           interp != ck_fp8_interpretation_t::CK_E5M2_OCP)                         \
        {                                                                          \
            __hip_assert(false && "type is unsupported by current target device"); \
        }                                                                          \
    }
#define __assert_fnuz_support(interp)                                              \
    {                                                                              \
        if(interp != ck_fp8_interpretation_t::CK_E4M3_FNUZ &&                      \
           interp != ck_fp8_interpretation_t::CK_E5M2_FNUZ)                        \
        {                                                                          \
            __hip_assert(false && "type is unsupported by current target device"); \
        }                                                                          \
    }

__host__ __device__ static inline void
__is_interpret_supported([[maybe_unused]] ck_fp8_interpretation_t interp)
{
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if CK_USE_OCP_FP8
    __assert_ocp_support(interp);
#endif
#if CK_USE_FNUZ_FP8
    __assert_fnuz_support(interp);
#endif
#endif
}

#if CK_FP8_CVT_FAST_PATH
// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_float8.h#L79
template <ck_fp8_interpretation_t interpret, bool saturate, bool stochastic_rounding = false>
static __device__ fp8_storage_t cast_to_f8_from_f32(float v, unsigned int rng = 0)
{
    fp8_storage_t i8data;
    union
    {
        float fval;
        unsigned int i32val;
        unsigned char i8val[4]; // NOTE: not endian independent
    } val;

    unsigned int ival = 0;
    val.fval          = v;

    if constexpr(saturate)
    {
        if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_FNUZ)
        {
            if((val.i32val & 0x7F800000) != 0x7F800000)
            { /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
            }
        }
        else if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
        { // OCP type
            if((val.i32val & 0x7F800000) != 0x7F800000)
            { /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 448.0, -448.0);
            }
        }
        else
        {
            if((val.i32val & 0x7F800000) != 0x7F800000)
            { /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
            }
        }
    }

    if constexpr(stochastic_rounding)
    {
        ival       = (interpret == ck_fp8_interpretation_t::CK_E4M3_FNUZ) ||
                       (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
                         ? __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0)
                         : __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
        val.i32val = ival;
        i8data     = val.i8val[0]; // little endian
    }
    else
    { // RNE CVT
        ival       = (interpret == ck_fp8_interpretation_t::CK_E4M3_FNUZ) ||
                       (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
                         ? __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false)
                         : __builtin_amdgcn_cvt_pk_bf8_f32(val.fval,
                                                     val.fval,
                                                     ival,
                                                     false); // false -> WORD0
        val.i32val = ival;
        i8data     = val.i8val[0];
    }
    return i8data;
}
#endif // CK_FP8_CVT_FAST_PATH

// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_hip_f8_impl.h#L39
// This has been modified to add double types conversion as well
template <typename T, int wm, int we, bool is_fnuz, bool clip = false, bool stoch = false>
__host__ __device__ static inline fp8_storage_t cast_to_f8(T _x, unsigned int rng = 0)
{
    constexpr bool is_half   = __hip_internal::is_same<T, _Float16>::value;
    constexpr bool is_float  = __hip_internal::is_same<T, float>::value;
    constexpr bool is_double = __hip_internal::is_same<T, double>::value;
    static_assert(is_half || is_float || is_double,
                  "Only half, float and double can be cast to f8");

    constexpr int mfmt = (sizeof(T) == 8) ? 52 : ((sizeof(T) == 4) ? 23 : 10);

    using T_bitwise = typename conditional<
        sizeof(T) == 2,
        unsigned short int,
        typename conditional<sizeof(T) == 4, unsigned int, unsigned long long>::type>::type;
    T_bitwise x_bitwise = bit_cast<T_bitwise>(_x);

    unsigned long long x{x_bitwise};

    unsigned long long head, mantissa;
    int exponent, bias;
    unsigned int sign;
    unsigned long long fInf, mask;

    if constexpr(sizeof(T) == 8)
    {
        head     = x & 0xFFF0000000000000ull;
        mantissa = x & 0xFFFFFFFFFFFFFull;
        exponent = (head >> 52) & 0x7FF;
        sign     = head >> 63;
        bias     = 1023;
        fInf     = 0x7FF0000000000000ull;
        mask     = 0x7FFFFFFFFFFFFFFFull;
    }
    else if constexpr(sizeof(T) == 4)
    {
        head     = x & 0xFF800000;
        mantissa = x & 0x7FFFFF;
        exponent = (head >> 23) & 0xFF;
        sign     = head >> 31;
        bias     = 127;
        fInf     = 0x7F800000;
        mask     = 0x7FFFFFFF;
    }
    else
    {
        head     = x & 0xFC00;
        mantissa = x & 0x3FF;
        exponent = (head >> 10) & 0x1F;
        sign     = head >> 15;
        bias     = 15;
        fInf     = 0x7C00;
        mask     = 0x7FFF;
    }
    unsigned int signed_inf = 0;
    unsigned int nan        = 0;
    if constexpr(is_fnuz)
    {
        signed_inf = clip ? ((sign << 7) + 0x7f) : 0x80;
        nan        = 0x80;
    }
    else
    {
        if constexpr(we == 4)
        { // e4m3
            signed_inf = (sign << 7) + (clip ? 0x7e : 0x7f);
        }
        else
        { // e5m2
            signed_inf = (sign << 7) + (clip ? 0x7b : 0x7c);
        }
        nan = (sign << 7) + 0x7f;
    }
    // Max values
    unsigned long long ifmax = 0;
    if constexpr(sizeof(T) == 8)
    {
        if constexpr(we == 5)
        { // 57344
            ifmax = 0x40EC000000000000ull;
        }
        else
        {
            if constexpr(is_fnuz)
            { // 240
                ifmax = 0x406E000000000000ull;
            }
            else
            { // 448
                ifmax = 0x407C000000000000ull;
            }
        }
    }
    else if(sizeof(T) == 4)
    {
        if constexpr(we == 5)
        {
            ifmax = 0x47600000;
        }
        else
        {
            if constexpr(is_fnuz)
            {
                ifmax = 0x43700000;
            }
            else
            {
                ifmax = 0x43E00000;
            }
        }
    }
    else
    {
        if constexpr(we == 5)
        {
            ifmax = 0x7B00;
        }
        else
        {
            if constexpr(is_fnuz)
            {
                ifmax = 0x5B80;
            }
            else
            {
                ifmax = 0x5F00;
            }
        }
    }
    // Deal with inf and NaNs
    if((x & fInf) == fInf)
    {
        if constexpr(is_fnuz)
            return signed_inf;

        return mantissa != 0 ? nan : signed_inf;
    }

    if((x & mask) > ifmax)
    {
        return signed_inf;
    }

    if(x == 0)
    {
        return 0;
    }

    // First need to check if it is normal or denorm as there is a difference of
    // implicit 1 Then need to adjust the exponent to align with the F8 exponent,
    // in the meanwhile, shift The mantissa. Then for stochastic rounding, add rng
    // to mantissa and truncate. And for RNE, no need to add rng. Then probably
    // need to check whether there is carry and adjust exponent and mantissa again

    // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent
    // bits
    const int f8_bias                  = (1 << (we - 1)) - 1 + (is_fnuz ? 1 : 0);
    const int f8_denormal_act_exponent = 1 - f8_bias; // actual exponent of f8 denormal
    // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
    // f8_exponent is the converted f8 exponent with bias encoding
    // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
    // the difference needs to be adjusted and mantissa shifted
    int act_exponent, f8_exponent, exponent_diff;

    if(exponent == 0)
    { // fp32/fp16 is in denormal.
        /* fp32 denormal is below 2^-127 so it is usually not a concern here, we
    mostly concern fp16 here. In this case, f8 is usually in denormal. But there
    could be exceptions. fp16 denormal has exponent bias 15 while bf8 with NANOO has
    exponent bias 16. It means that there are some numbers in fp16 denormal but they
    are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. fp16 numbers
    where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8
    (NANOO) normal. In this case, the fp16 mantissa should be shift left by 1  */
        act_exponent  = exponent - bias + 1;
        exponent_diff = f8_denormal_act_exponent -
                        act_exponent; // actual exponent is exponent-bias+1 as it is denormal
    }
    else
    { // fp32/fp16 is normal with implicit 1
        act_exponent = exponent - bias;
        if(act_exponent <= f8_denormal_act_exponent)
        {
            /* This is the case where fp32/fp16 is normal but it is in f8 denormal
      range. For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
      actual exponent is -7, it is actually larger due to the implicit 1,
      Therefore it needs to be adjust to -6 and mantissa shift right by 1.
      So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
            exponent_diff = f8_denormal_act_exponent - act_exponent;
        }
        else
        {                      // both fp32/fp16 and f8 are in normal range
            exponent_diff = 0; // exponent_diff=0 does not mean there is no difference
                               // for this case, act_exponent could be larger. Just
                               // that it does not need shift mantissa
        }
        mantissa += (1ull << mfmt); // Add the implicit 1 into mantissa
    }

    bool midpoint = (mantissa & ((1ull << (mfmt - wm + exponent_diff)) - 1)) ==
                    (1ull << (mfmt - wm + exponent_diff - 1));
    /* This part is a bit tricky. The judgment of whether it is a tie needs to be
  done before we shift right as shift right could rip off some residual part and
  make something not midpoint look like midpoint. For example, the fp16 number
  0x1002 (0 00100 0000000010), it is larger than midpoint, but after shift right
  by 4 bits, it would look like midpoint.
  */

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1ull << mfmt);
    // if there is no implicit 1, it  means the f8 is denormal and need to adjust
    // to denorm exponent
    f8_exponent =
        (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias - (implicit_one ? 0 : 1);

    // Now we have the exponent and mantissa adjusted
    unsigned long long drop_mask = (1ull << (mfmt - wm)) - 1;
    bool odd =
        mantissa & (1ull << (mfmt - wm)); // if the least significant bit that is not truncated is 1
    mantissa +=
        (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1ull) : mantissa)) & drop_mask;

    // Now we deal with overflow
    if(f8_exponent == 0)
    {
        if((1ull << mfmt) & mantissa)
        {
            f8_exponent = 1; // denormal overflow to become normal, promote exponent
        }
    }
    else
    {
        if((1ull << (mfmt + 1)) & mantissa)
        {
            mantissa >>= 1;
            f8_exponent++;
        }
    }

    mantissa >>= (mfmt - wm);

    // above range: quantize to maximum possible float of the same sign
    const int max_exp = (1 << we) - 1;
    if(f8_exponent > max_exp)
    {
        if constexpr(clip)
        {
            mantissa    = (1 << wm) - 1;
            f8_exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }

    if(f8_exponent == 0 && mantissa == 0)
        return is_fnuz ? 0 : (sign << 7);
    mantissa &= (1 << wm) - 1;
    return (sign << 7) | (f8_exponent << wm) | mantissa;
}

/**
 * \brief convert float to @p fp8_storage_t
 *
 * \tparam interp interpretation of fp8
 * \tparam sat saturation of fp8
 * \param f float number
 * \return fp8_storage_t
 */
template <ck_fp8_interpretation_t interp,
          ck_saturation_t sat      = ck_saturation_t::CK_SATFINITE,
          bool stochastic_rounding = false>
#if CK_FP8_CVT_FAST_PATH
__host__ __device__ static inline fp8_storage_t cvt_float_to_fp8(const float f)
{
    __is_interpret_supported(interp);
    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
        constexpr int seed = 1254739;
        rng                = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&f), f);
    }
    return cast_to_f8_from_f32<interp, sat == ck_saturation_t::CK_SATFINITE, stochastic_rounding>(
        f, rng);
#else
#if CK_USE_OCP_FP8
__host__ __device__ static inline fp8_storage_t cvt_float_to_fp8(const float f)
{
#else
__host__ static inline fp8_storage_t cvt_float_to_fp8(const float f)
{
#endif
    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
        constexpr int seed = 1254739;
        rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&f), f);
    }

    if constexpr(interp == ck_fp8_interpretation_t::CK_E4M3_FNUZ)
    {
        return cast_to_f8<float,
                          3,
                          4,
                          true,
                          sat == ck_saturation_t::CK_SATFINITE,
                          stochastic_rounding>(f, rng);
    }
    else if constexpr(interp == ck_fp8_interpretation_t::CK_E5M2_FNUZ)
    {
        return cast_to_f8<float,
                          2,
                          5,
                          true,
                          sat == ck_saturation_t::CK_SATFINITE,
                          stochastic_rounding>(f, rng);
    }
    else if constexpr(interp == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return cast_to_f8<float,
                          3,
                          4,
                          false,
                          sat == ck_saturation_t::CK_SATFINITE,
                          stochastic_rounding>(f, rng);
    }
    else if constexpr(interp == ck_fp8_interpretation_t::CK_E5M2_OCP)
    {
        return cast_to_f8<float,
                          2,
                          5,
                          false,
                          sat == ck_saturation_t::CK_SATFINITE,
                          stochastic_rounding>(f, rng);
    }
    else
    {
        __hip_assert(false && "FP8 type is not supported by current target device");
        return 0;
    }
#endif // CK_FP8_CVT_FAST_PATH
}

/**
 * \brief convert _Float16 to @p fp8_storage_t
 *
 * \tparam sat saturation of fp8
 * \tparam interp interpretation of fp8
 * \tparam stochastic_rounding switch between RNE and SR
 * \param x _Float16 value
 * \return fp8_storage_t
 */
template <ck_fp8_interpretation_t interp,
          ck_saturation_t sat      = ck_saturation_t::CK_SATFINITE,
          bool stochastic_rounding = false>
#if CK_FP8_CVT_FAST_PATH || CK_USE_OCP_FP8
__host__ __device__ static inline fp8_storage_t cvt_half_t_to_fp8(const _Float16 x)
#else
__host__ static inline fp8_storage_t cvt_half_t_to_fp8(const _Float16 x)
#endif
{
    return cvt_float_to_fp8<interp, sat, stochastic_rounding>(static_cast<float>(x));
}

} // namespace fp8_impl

// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_rne(X x);

// convert fp32 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, float>(float x)
{
    return f8_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

// convert fp32 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_rne<bf8_ocp_t, float>(float x)
{
    return bf8_ocp_t{
        fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(x)};
}

// convert _Float16 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, _Float16>(_Float16 x)
{
    return f8_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

template <>
inline __host__ __device__ bf8_ocp_t f8_convert_rne<bf8_ocp_t, _Float16>(_Float16 x)
{
    return bf8_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(
            x)};
}

// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_sr(X x);

// convert fp32 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, float>(float x)
{
    return f8_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation, true>(
            x)};
}

// convert fp32 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_sr<bf8_ocp_t, float>(float x)
{
    return bf8_ocp_t{fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret,
                                                bf8_ocp_t::default_saturation,
                                                true>(x)};
}

// convert _Float16 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, _Float16>(_Float16 x)
{
    return f8_ocp_t{fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret,
                                                f8_ocp_t::default_saturation,
                                                true>(x)};
}

// convert _Float16 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_sr<bf8_ocp_t, _Float16>(_Float16 x)
{
    return bf8_ocp_t{fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret,
                                                 bf8_ocp_t::default_saturation,
                                                 true>(x)};
}

#if CK_USE_OCP_FP8
using f8_t  = f8_ocp_t;
using bf8_t = bf8_ocp_t;
#define CK_FP8_TYPE_FNUZ 0
#define CK_FP8_TYPE_OCP 1
#else
using f8_t = f8_fnuz_t;
using bf8_t = bf8_fnuz_t;
#define CK_FP8_TYPE_FNUZ 1
#define CK_FP8_TYPE_OCP 0
#endif

} // namespace ck
