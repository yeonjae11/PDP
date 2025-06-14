# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Qwen model implementation for torchtitan

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .model import Transformer, TransformerModelArgs
from .parallelize_qwen import parallelize_qwen
from .pipeline_qwen import pipeline_qwen

__all__ = [
    "parallelize_qwen",
    "pipeline_qwen",
    "TransformerModelArgs",
    "Transformer",
    "qwen2_configs",
]


qwen2_configs = {
    "debugmodel": TransformerModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "medium": TransformerModelArgs(dim=2048, n_layers=24, n_heads=16, rope_theta=500000),
    "1.5B": TransformerModelArgs(
        dim=2048, 
        n_layers=24, 
        n_heads=16, 
        n_kv_heads=16, 
        norm_eps=1e-6, 
        rope_theta=500000,
        max_seq_len=4096,
    ),
    "7B": TransformerModelArgs(
        dim=3584, 
        n_layers=28, 
        n_heads=28, 
        n_kv_heads=4, 
        intermediate_size=18944,
        norm_eps=1e-6, 
        rope_theta=1000000.0,
        max_seq_len=4096,
    ),
    "32B": TransformerModelArgs(
        dim=5120, 
        n_layers=64, 
        n_heads=40, 
        n_kv_heads=8, 
        intermediate_size=27648,
        norm_eps=1e-5, 
        rope_theta=1000000.0,
        max_seq_len=4096,
    ),
    "72B": TransformerModelArgs(
        dim=8192, 
        n_layers=80, 
        n_heads=64, 
        n_kv_heads=8, 
        intermediate_size=29568,
        norm_eps=1e-5, 
        rope_theta=1000000.0,
        max_seq_len=4096,
    ),
}

register_train_spec(
    TrainSpec(
        name="qwen2",
        cls=Transformer,
        config=qwen2_configs,
        parallelize_fn=parallelize_qwen,
        pipelining_fn=pipeline_qwen,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
