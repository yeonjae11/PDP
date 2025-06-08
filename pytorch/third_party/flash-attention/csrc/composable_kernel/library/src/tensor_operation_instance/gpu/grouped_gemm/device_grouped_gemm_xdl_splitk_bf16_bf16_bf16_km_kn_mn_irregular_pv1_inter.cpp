// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_km_kn_mn_irregular_pv1_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  BF16,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_splitk_2Bt_crr_instances<BF16,
                                                         GemmMNKPadding,
                                                         PipelineV1,
                                                         InterwaveScheduler>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
