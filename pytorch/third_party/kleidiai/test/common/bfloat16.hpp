//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <type_traits>

#include "test/common/type_traits.hpp"

namespace kai::test {

/// Half-precision brain floating-point.
class BFloat16 {
public:
    /// Constructor.
    BFloat16() = default;

    /// Creates a new object from the specified numeric value.
    explicit BFloat16(float value) : m_data(float_to_bfloat16_round_towards_zero(value)) {
    }

    /// Assigns to the specified numeric value which will be converted to `bfloat16_t`.
    template <typename T, std::enable_if_t<is_arithmetic<T>, bool> = true>
    BFloat16& operator=(T value) {
        const auto value_f32 = static_cast<float>(value);
        m_data = float_to_bfloat16_round_towards_zero(value_f32);
        return *this;
    }

    /// Converts to single-precision floating-point.
    explicit operator float() const {
        float value_f32 = 0.0F;
        uint32_t value_u32 = static_cast<uint32_t>(m_data) << 16;

        memcpy(&value_f32, &value_u32, sizeof(float));

        return value_f32;
    }

private:
    /// Equality operator.
    [[nodiscard]] friend bool operator==(BFloat16 lhs, BFloat16 rhs) {
        return lhs.m_data == rhs.m_data;
    }

    /// Inequality operator.
    [[nodiscard]] friend bool operator!=(BFloat16 lhs, BFloat16 rhs) {
        return lhs.m_data != rhs.m_data;
    }

    /// Writes the value to the output stream.
    ///
    /// @param[in] os Output stream to be written to.
    /// @param[in] value Value to be written.
    ///
    /// @return The output stream.
    friend std::ostream& operator<<(std::ostream& os, BFloat16 value);

    static uint16_t float_to_bfloat16_round_towards_zero(float value);

    uint16_t m_data;
};

}  // namespace kai::test
