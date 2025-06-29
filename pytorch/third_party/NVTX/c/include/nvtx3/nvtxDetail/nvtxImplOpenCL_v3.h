/*
 * SPDX-FileCopyrightText: Copyright (c) 2009-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://nvidia.github.io/NVTX/LICENSE.txt for license information.
 */

#ifndef NVTX_IMPL_GUARD_OPENCL
#error Never include this file directly -- it is automatically included by nvToolsExtCuda.h (except when NVTX_NO_IMPL is defined).
#endif

#if defined(NVTX_AS_SYSTEM_HEADER)
#if defined(__clang__)
#pragma clang system_header
#elif defined(__GNUC__) || defined(__NVCOMPILER)
#pragma GCC system_header
#elif defined(_MSC_VER)
#pragma system_header
#endif
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void (NVTX_API * nvtxNameClDeviceA_impl_fntype)(cl_device_id device, const char* name);
typedef void (NVTX_API * nvtxNameClDeviceW_impl_fntype)(cl_device_id device, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClContextA_impl_fntype)(cl_context context, const char* name);
typedef void (NVTX_API * nvtxNameClContextW_impl_fntype)(cl_context context, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClCommandQueueA_impl_fntype)(cl_command_queue command_queue, const char* name);
typedef void (NVTX_API * nvtxNameClCommandQueueW_impl_fntype)(cl_command_queue command_queue, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClMemObjectA_impl_fntype)(cl_mem memobj, const char* name);
typedef void (NVTX_API * nvtxNameClMemObjectW_impl_fntype)(cl_mem memobj, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClSamplerA_impl_fntype)(cl_sampler sampler, const char* name);
typedef void (NVTX_API * nvtxNameClSamplerW_impl_fntype)(cl_sampler sampler, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClProgramA_impl_fntype)(cl_program program, const char* name);
typedef void (NVTX_API * nvtxNameClProgramW_impl_fntype)(cl_program program, const wchar_t* name);
typedef void (NVTX_API * nvtxNameClEventA_impl_fntype)(cl_event evnt, const char* name);
typedef void (NVTX_API * nvtxNameClEventW_impl_fntype)(cl_event evnt, const wchar_t* name);

NVTX_DECLSPEC void NVTX_API nvtxNameClDeviceA(cl_device_id device, const char* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)device;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClDeviceA_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClDeviceA_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClDeviceA_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(device, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClDeviceW(cl_device_id device, const wchar_t* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)device;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClDeviceW_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClDeviceW_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClDeviceW_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(device, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClContextA(cl_context context, const char* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)context;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClContextA_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClContextA_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClContextA_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(context, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClContextW(cl_context context, const wchar_t* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)context;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClContextW_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClContextW_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClContextW_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(context, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClCommandQueueA(cl_command_queue command_queue, const char* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)command_queue;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClCommandQueueA_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClCommandQueueA_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClCommandQueueA_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(command_queue, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClCommandQueueW(cl_command_queue command_queue, const wchar_t* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)command_queue;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClCommandQueueW_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClCommandQueueW_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClCommandQueueW_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(command_queue, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClMemObjectA(cl_mem memobj, const char* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)memobj;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClMemObjectA_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClMemObjectA_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClMemObjectA_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(memobj, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClMemObjectW(cl_mem memobj, const wchar_t* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)memobj;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClMemObjectW_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClMemObjectW_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClMemObjectW_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(memobj, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClSamplerA(cl_sampler sampler, const char* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)sampler;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClSamplerA_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClSamplerA_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClSamplerA_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(sampler, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClSamplerW(cl_sampler sampler, const wchar_t* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)sampler;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClSamplerW_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClSamplerW_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClSamplerW_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(sampler, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClProgramA(cl_program program, const char* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)program;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClProgramA_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClProgramA_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClProgramA_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(program, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClProgramW(cl_program program, const wchar_t* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)program;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClProgramW_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClProgramW_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClProgramW_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(program, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClEventA(cl_event evnt, const char* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)evnt;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClEventA_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClEventA_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClEventA_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(evnt, name);
#endif /* NVTX_DISABLE */
}

NVTX_DECLSPEC void NVTX_API nvtxNameClEventW(cl_event evnt, const wchar_t* name)
{
    NVTX_SET_NAME_MANGLING_OPTIONS
#ifdef NVTX_DISABLE
    (void)evnt;
    (void)name;
#else /* NVTX_DISABLE */
    nvtxNameClEventW_impl_fntype local = NVTX_REINTERPRET_CAST(nvtxNameClEventW_impl_fntype, NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameClEventW_impl_fnptr);
    if (local != NVTX_NULLPTR)
        (*local)(evnt, name);
#endif /* NVTX_DISABLE */
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
