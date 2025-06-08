<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Changelog

KleidiAI follows the [Semantic Versioning](https://semver.org/) specification for releases.

## Upcoming Release

## v1.8.0

- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN) Micro-kernels of QAI8DX LHS and QSI8CX RHS with F16 output, optimized for FEAT_I8MM and FEAT_DotProd.
  - Matrix multiplication (1xN) Micro-kernels of QAI8DX LHS and QSI8CX RHS with F16 output, optimized for FEAT_DotProd.
- New SME micro-kernels:
  - Indirect matrix multiplication (MxN) of F16 input and output.
    - Packing kernels for LHS and RHS
  - Indirect matrix multiplication (MxN) of F32 input and output.
    - Packing kernels for LHS and RHS
- New SME2 micro-kernels:
  - Indirect matrix multiplication (MxN) of F16 input and output.
    - Matrix multiplication of packed indirect LHS and packed RHS
  - Indirect matrix multiplication (MxN) of F32 input and output.
    - Matrix multiplication of packed indirect LHS and packed RHS
- Disable link time optimization for microkernel library

## v1.7.0

- New SME micro-kernels:
  - Indirect matrix multiplication (MxN) of QAI8 input and output.
    - Packing kernels for LHS and RHS
- New SME2 micro-kernels:
  - Indirect matrix multiplication (MxN) of QAI8 input and output.
    - Matrix multiplication of packed indirect LHS and packed RHS
- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F32 output, optimized for FEAT_I8MM.
  - Matrix multiplication (1xN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F32 output, optimized for FEAT_DotProd.
  - Matrix multiplication (MxN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F16 output, optimized for FEAT_I8MM.
  - Matrix multiplication (1xN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F16 output, optimized for FEAT_DotProd.
  - Matrix multiplication (MxN) Micro-kernels of QAI8DX LHS and QSI4CX RHS with F16 output, optimized for FEAT_I8MM and FEAT_DotProd.
  - Matrix multiplication (1xN) Micro-kernels of QAI8DX LHS and QSI4CX RHS with F16 output, optimized for FEAT_DotProd.

## v1.6.0

- Add CMake installation and `find_package()` support.
- Optimize RHS packing qsu4c32s16s0->qsi4c32pscalef16
- Fixes:
  - Fix issue where the following kernels ignored clamping parameters:
    - kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla
    - kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot
    - kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla

## v1.5.0

- Extend benchmark tool to support all matrix multiplication micro-kernels.
- New Advanced SIMD micro-kernels:
  - New 4x8 block size variant of matrix multiplication of QAI8DXP LHS and QSI4C32P RHS with F32 output.
    - Optimizations for FEAT_I8MM.
- Fixes:
  - Remove "-Weffc++" from build flags
  - Fix out-of-bound read from LHS packed matrix in `kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa`.

## v1.4.0

- New Advanced SIMD micro-kernels:
  - New 4x8 block size variant of matrix multiplication of QAI8DXP LHS and QSI4C32P RHS with F32 output.
    - Optimizations for FEAT_DotProd.
  - New 1x8 block size variant of matrix multiplication of QAI8DXP LHS and QSI4C32P RHS with F32 output.
    - Optimizations for FEAT_DotProd.
  - New 1x8 block size variant of matrix multiplication of QAI8DXP 1x8 LHS and QSI4C32P 8x8 RHS with F32 output.
    - Optimizations for FEAT_DotProd.
- New SME2 micro-kernels:
  - Matrix multiplication (1xN) of QAI8 LHS and QSI8 RHS to produce QAI8 output.
- Updated an example to demonstrate integration using CMake
- Build tests for matmul_clamp_f32_qai8dxp_qsi4c32p with MSVC
- Fixes:
  - Fix the RHS packing micro-kernel kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon to handle null bias.
  - Implement matmul portion testing in int8 unit tests
  - Use absolute path as header search path in CMakeLists.txt

## v1.3.0

- Update FP16 example to use NHWC input
- Fixes:
  - Fix build error on MSVC for some kai_matmul_clamp_f32_qai8dxp_qsi4c32p micro-kernels
  - Fix compilation warnings detected by `-Wcast-qual -Wmissing-prototypes -Wstrict-prototypes -Woverlength-strings` compiler options.
    - Support compiling the project with the above compilation options enabled.
  - Remove `-Werror` from default build flags as to not cause integration problems
  - Expose the rhs_packed_stride in the header file
  - Fix validation error when n > nr in kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa

## v1.2.0

- New SME micro-kernels:
  - Matrix multiplication (MxN) for BF16 inputs with F32 output.
- Add MSVC support for test framework
- Fixes:
  - Fix several CPU feature check issues affecting test framework
  - Fix the LHS/RHS packed offset calculation in matmul get_offset methods

## v1.1.0

- New Advanced SIMD micro-kernels:
  - New 16x4 and 1x4 block size variants of matrix multiplication of QAI8DXP LHS and QSI4C32P RHS with F32 output.
    - Optimizations for FEAT_DotProd.
- New SME micro-kernels:
  - Matrix multiplication (MxN and 1xN) of QAI8DXP LHS and QSI4CXP RHS to produce F32 output.
- Packing micro-kernels for QSI4CXP RHS to work with the SME matrix multiplication (MxN and 1xN) micro-kernels.
- Fixes:
  - Fix out-of-bounds read in `kai_lhs_quant_pack_qai8dxp_f32` packing micro-kernel.
  - Unit test improvements.

## v1.0.0

- Breaking changes:
  - Change the F16 matrix multiplication function signature to use single-precision floating-point for the clamp values.
- Optimizations:
  - Optimize QAI8DXP LHS quant and pack micro-kernel using Arm® Neon™
  - Optimize the NxK scalar RHS packing function for QSU4C32 with BF16 quantization scales
- Add initial Microsoft® Visual C++™ build support
- API for querying library version
- Fixes:
  - Update QSI8CX tests
  - Asserts will call `abort()` instead of `exit(...)`
  - Changed invalid assertion in F16 kernel
  - Build system improvements
  - Unit test improvements

## v0.5.0

- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN and 1xN) of QSI8D32 LHS (dynamic 8-bit integer per-block quantized) and QSI4C32 RHS (4-bit integer per-block quantized) to produce F32 output.
    - Optimizations for FEAT_DotProd.
  - Matrix multiplication (MxN and 1xN) of QAI8DX LHS (dynamic 8-bit integer per-row quantized) and QSI4CX RHS (4-bit integer per-channel quantized) to produce F32 output.
    - Optimizations for FEAT_DotProd and FEAT_I8MM.
    - Packing micro-kernels for LHS and non-transposed and transposed RHS.
  - Matrix multiplication (MxN) of BF16 LHS and BF16 RHS to produce F16 output.
    - Packing micro-kernels for LHS and non-transposed RHS.
- New SME micro-kernels:
  - Matrix multiplication (MxN and 1xN) of F16 LHS and F16 RHS to produce F16 output.
    - Packing micro-kernels for LHS and non-transposed and transposed RHS.
  - Matrix multiplication (MxN) of QAI8 LHS and QSI8 RHS to produce QAI8 output.
    - Packing micro-kernels for LHS and non-transposed RHS.
  - Matrix multiplication (MxN and 1xN) of QSI8D32 LHS and QSI4C32 RHS to produce F32 output
- Packing micro-kernels for QSI8D32 LHS and non-transposed QSI4C32 RHS, to work with the SME matrix multiplication (MxN and 1xN) micro-kernels.
- Fixes:
  - Fixes relating to illegal instruction errors on systems with SME but without SVE support:
    - Contain SME assembly inside the SMSTART and SMSTOP boundary.
    - Disable compiler generated SVE instructions by adding the -fno-tree-vectorize compiler option to the build.
  - Fix build warnings in the core library introduced by the -Wpedantic compiler option.
  - Fix typos in the micro-kernel interface files.

## v0.4.0

- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN) of QAI8DX (dynamically quantized 8-bit integer) LHS and QSI4CX (quantized 4-bit integer) RHS with F32 output.
  - Matrix multiplication (MxN and 1xN) of BF16 LHS and RHS with F32 output.
- New SME micro-kernels:
  - SME2 F32 matrix multiplication (1xN) micro-kernels:
    - Compatible with 2VL RHS packing, for sharing one packed RHS with SME2 F32 GEMM micro-kernel.
    - Compatible with 16VL RHS packing.
  - SME F32 packing function for transposed RHS matrix.
- Enhancements to existing micro-kernels:
  - Port several quantized micro-kernels to optimized Advanced SIMD assembly.
- Register SME F32 matrix multiplication micro-kernel in the benchmark suite.
- Enable air gapped CMake builds through local third-party dependencies.

## v0.3.0

- Advanced SIMD FP32 GEMM micro-kernel.
- Micro-kernels to compute the matrix multiplication of dynamically quantized asymmetric signed 8-bit integer with per-row quantization (QAI8DX) LHS and quantized symmetric 4-bit signed integer with per-block quantization (QSI4C32) RHS. The destination matrix data type is single-precision floating-point (F32). The micro-kernels have been optimized using the Arm® CPU feature FEAT_I8MM for the matrix-by-matrix cases and the FEAT_DotProd for the vector-by-matrix cases.
- RHS matrix packing micro-kernels to pack the RHS matrix holding the QSI4C32 values.
- Unit test and example for integer micro-kernels.
- Extend support for signed 4-bit integer inputs in quantized symmetric 4-bit signed integer with per-channel quantization (QSI4CXP) RHS packing micro-kernel.
  - kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0 renamed to kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.
  - kai_rhs_pack_kxn_qsi4cxp_qsu4cxs1s0 renamed to kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.
- Remove FP16 GEMV micro-kernel optimized for Advanced SIMD.
  - Where a dedicated GEMV micro-kernel is not provided, it is recommended to use existing GEMM micro-kernels which have dedicated paths for M=1 (a "GEMV" operation).

## v0.2.0

- Micro-kernels to compute the matrix multiplication of dynamically quantized symmetric signed 8-bit integer with
  per-block quantization (QSI8D32) activations and quantized symmetric 4-bit signed integer with per-block quantization
  (QSI4C32) weights and the accumulation of the result into a single-precision (F32) output,
  optimized for Arm® Neon™ technology.
- Tensor packing micro-kernels to prepare the activations and weights for input to the above matrix multiplication
  micro-kernel.
- Unit test and example for integer micro-kernels.

## v0.1.0

The first release of KleidiAI includes:

- Micro-kernels to compute the matrix multiplication of:
  - Dynamically quantized 8-bit integer (QAI8DX) activations and quantized 4-bit integer (QSI4CX) weights and the
    accumulation of the result into a single-precision (F32) output, optimized for Arm® Neon™ technology.
  - Half precision floating-point (F16) activations and weights and the accumulation of the result into an F16 output,
    optimized for Neon technology.
  - F32 activations and weights and the accumulation of the result into an F32 output, optimized for SME2 technology.
- Tensor packing micro-kernels to prepare the activations and weights for input to the above matrix multiplication
  micro-kernels.
- Examples and documentation demonstrating the usage of the 4-bit integer and 16-bit floating point matrix
  multiplication micro-kernels.
- Testing suite.
- CMake and Bazel build system for micro kernels.
