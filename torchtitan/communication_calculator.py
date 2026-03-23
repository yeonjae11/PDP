import math
import torch
import argparse
from typing import Dict, Tuple, List, Optional

def calculate_tp_communication_volume(
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    seq_length: int,
    batch_size: int,
    tp_size: int,
    dtype_size: int = 2,  # Default to FP16/BF16 (2 bytes)
    include_backward: bool = True,
):
    """
    Calculate communication volume for tensor parallelism in Llama model.
    
    Args:
        hidden_size: Model hidden dimension size
        intermediate_size: Size of intermediate feed-forward dimension
        num_attention_heads: Number of attention heads
        seq_length: Sequence length
        batch_size: Batch size
        tp_size: Tensor parallel size (number of GPUs)
        dtype_size: Size of data type in bytes (default: 2 for FP16/BF16)
        include_backward: Whether to include backward pass communication (default: True)
        
    Returns:
        Dictionary with communication volume breakdown and total
    """
    # Validate inputs
    if tp_size <= 1:
        return {"total_bytes": 0, "message": "No communication needed for tp_size=1"}
    
    if tp_size > hidden_size or tp_size > intermediate_size:
        raise ValueError("TP size cannot be larger than hidden or intermediate dimensions")

    comm_volume = {}
    total_bytes = 0
    
    # Size per sample (B×S = batch_size × seq_length)
    bs_size = batch_size * seq_length
    
    # ===== Forward Pass Communication =====
    
    # 1. Embedding layer (RowwiseParallel)
    # No communication in forward for embedding (RowwiseParallel)
    
    # 2. Attention layers
    # ColwiseParallel for Q,K,V matrices (ReduceScatter in forward)
    # Each Q,K,V processes hidden_size/tp_size → hidden_size sized input
    # and produces output of reduced size
    qkv_forward_rs_bytes = 3 * bs_size * hidden_size * dtype_size
    comm_volume["attention_qkv_forward"] = qkv_forward_rs_bytes
    total_bytes += qkv_forward_rs_bytes
    
    # RowwiseParallel for output projection (AllGather in forward)
    # Each output projection processes hidden_size/tp_size sized input locally
    # and needs to gather results across TP dimension
    wo_forward_ag_bytes = bs_size * hidden_size * dtype_size
    comm_volume["attention_output_forward"] = wo_forward_ag_bytes
    total_bytes += wo_forward_ag_bytes
    
    # 3. Feed Forward layers
    # w1 and w3: ColwiseParallel (ReduceScatter in forward)
    w1_w3_forward_rs_bytes = 2 * bs_size * hidden_size * dtype_size
    comm_volume["ffn_w1_w3_forward"] = w1_w3_forward_rs_bytes
    total_bytes += w1_w3_forward_rs_bytes
    
    # w2: RowwiseParallel (AllGather in forward)
    w2_forward_ag_bytes = bs_size * intermediate_size * dtype_size
    comm_volume["ffn_w2_forward"] = w2_forward_ag_bytes
    total_bytes += w2_forward_ag_bytes
    
    # ===== Backward Pass Communication =====
    if include_backward:
        # 1. Embedding layer (RowwiseParallel)
        # ReduceScatter in backward for embedding
        embed_backward_rs_bytes = bs_size * hidden_size * dtype_size
        comm_volume["embedding_backward"] = embed_backward_rs_bytes
        total_bytes += embed_backward_rs_bytes
        
        # 2. Attention layers
        # ColwiseParallel for Q,K,V (AllGather in backward)
        qkv_backward_ag_bytes = 3 * bs_size * hidden_size * dtype_size
        comm_volume["attention_qkv_backward"] = qkv_backward_ag_bytes
        total_bytes += qkv_backward_ag_bytes
        
        # RowwiseParallel for output projection (ReduceScatter in backward)
        wo_backward_rs_bytes = bs_size * hidden_size * dtype_size
        comm_volume["attention_output_backward"] = wo_backward_rs_bytes
        total_bytes += wo_backward_rs_bytes
        
        # 3. Feed Forward layers
        # w1 and w3: ColwiseParallel (AllGather in backward)
        w1_w3_backward_ag_bytes = 2 * bs_size * hidden_size * dtype_size 
        comm_volume["ffn_w1_w3_backward"] = w1_w3_backward_ag_bytes
        total_bytes += w1_w3_backward_ag_bytes
        
        # w2: RowwiseParallel (ReduceScatter in backward)
        w2_backward_rs_bytes = bs_size * intermediate_size * dtype_size
        comm_volume["ffn_w2_backward"] = w2_backward_rs_bytes
        total_bytes += w2_backward_rs_bytes
    
    # Add total for each direction
    forward_bytes = qkv_forward_rs_bytes + wo_forward_ag_bytes + w1_w3_forward_rs_bytes + w2_forward_ag_bytes
    comm_volume["forward_total_bytes"] = forward_bytes
    
    if include_backward:
        backward_bytes = (embed_backward_rs_bytes + qkv_backward_ag_bytes + wo_backward_rs_bytes + 
                        w1_w3_backward_ag_bytes + w2_backward_rs_bytes)
        comm_volume["backward_total_bytes"] = backward_bytes
    
    # Calculate per layer numbers (for one transformer block)
    comm_volume["per_layer_forward_bytes"] = qkv_forward_rs_bytes + wo_forward_ag_bytes + w1_w3_forward_rs_bytes + w2_forward_ag_bytes
    
    if include_backward:
        comm_volume["per_layer_backward_bytes"] = qkv_backward_ag_bytes + wo_backward_rs_bytes + w1_w3_backward_ag_bytes + w2_backward_rs_bytes
        comm_volume["per_layer_total_bytes"] = comm_volume["per_layer_forward_bytes"] + comm_volume["per_layer_backward_bytes"]
    else:
        comm_volume["per_layer_total_bytes"] = comm_volume["per_layer_forward_bytes"]
    
    # Total volume
    comm_volume["total_bytes"] = total_bytes
    
    # Convert to more readable formats
    comm_volume["total_gb"] = total_bytes / (1024 ** 3)
    comm_volume["forward_gb"] = forward_bytes / (1024 ** 3)
    if include_backward:
        comm_volume["backward_gb"] = backward_bytes / (1024 ** 3)
    
    return comm_volume


def calculate_pipeline_communication_volume(
    hidden_size: int,
    num_layers: int,
    seq_length: int,
    batch_size: int,
    micro_batch_size: int,
    pp_size: int,
    dtype_size: int = 2,  # Default to FP16/BF16 (2 bytes)
):
    """
    Calculate communication volume for pipeline parallelism in Llama model.
    
    Args:
        hidden_size: Model hidden dimension size
        num_layers: Number of transformer layers
        seq_length: Sequence length
        batch_size: Global batch size
        micro_batch_size: Micro-batch size for pipeline parallelism
        pp_size: Pipeline parallel size (number of stages)
        dtype_size: Size of data type in bytes (default: 2 for FP16/BF16)
        
    Returns:
        Dictionary with communication volume breakdown and total
    """
    # Validate inputs
    if pp_size <= 1:
        return {"total_bytes": 0, "message": "No communication needed for pp_size=1"}
    
    if num_layers < pp_size:
        raise ValueError("Number of layers must be greater than or equal to PP size")
    
    comm_volume = {}
    
    # Number of micro batches
    num_micro_batches = batch_size // micro_batch_size
    if batch_size % micro_batch_size != 0:
        num_micro_batches = math.ceil(batch_size / micro_batch_size)
        print(f"Warning: Batch size {batch_size} not evenly divisible by micro batch size {micro_batch_size}. "
              f"Using {num_micro_batches} micro-batches.")
    
    # Size of activations sent between pipeline stages
    # Each pipeline boundary sends/receives activations of size (micro_batch_size × seq_length × hidden_size)
    activation_size = micro_batch_size * seq_length * hidden_size * dtype_size
    
    # In 1F1B, each micro-batch crosses (pp_size-1) pipeline boundaries in forward
    # and (pp_size-1) pipeline boundaries in backward
    total_forwards = num_micro_batches * (pp_size - 1)
    total_backwards = num_micro_batches * (pp_size - 1)
    
    forward_bytes = activation_size * total_forwards
    backward_bytes = activation_size * total_backwards
    
    comm_volume["forward_bytes"] = forward_bytes
    comm_volume["backward_bytes"] = backward_bytes
    comm_volume["total_bytes"] = forward_bytes + backward_bytes
    
    # Convert to more readable formats
    comm_volume["forward_gb"] = forward_bytes / (1024 ** 3)
    comm_volume["backward_gb"] = backward_bytes / (1024 ** 3)
    comm_volume["total_gb"] = comm_volume["total_bytes"] / (1024 ** 3)
    
    return comm_volume


def get_llama_model_config(model_size: str) -> Dict[str, int]:
    """
    Get configuration parameters for different Llama model sizes.
    
    Args:
        model_size: Size of the model (e.g., '7B', '13B', '70B', etc.)
        
    Returns:
        Dictionary with model configuration parameters
    """
    configs = {
        # Llama 3 8B parameters
        "8B": {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
        },
        # Llama 2 7B parameters
        "7B": {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
        },
        # Llama 2 13B parameters
        "13B": {
            "hidden_size": 5120,
            "intermediate_size": 13824,
            "num_attention_heads": 40,
            "num_hidden_layers": 40,
        },
        # Llama 2 70B parameters
        "70B": {
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_attention_heads": 64,
            "num_hidden_layers": 80,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size {model_size} not supported. Choose from: {list(configs.keys())}")
    
    return configs[model_size]


def main():
    parser = argparse.ArgumentParser(description="Calculate communication volume for parallel training")
    parser.add_argument("--model_size", default="8B", choices=["7B", "8B", "13B", "70B"], help="Model size")
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Global batch size")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=2, help="Pipeline parallel size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro-batch size for pipeline parallelism")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16", "int8"], help="Data type")
    
    args = parser.parse_args()
    
    # Map dtype to size in bytes
    dtype_sizes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
    }
    dtype_size = dtype_sizes[args.dtype]
    
    # Get model config
    model_config = get_llama_model_config(args.model_size)
    
    print(f"\n--- Communication Volume Analysis for Llama {args.model_size} ---")
    print(f"Model: Llama {args.model_size}")
    print(f"Hidden Size: {model_config['hidden_size']}")
    print(f"Intermediate Size: {model_config['intermediate_size']}")
    print(f"Attention Heads: {model_config['num_attention_heads']}")
    print(f"Layers: {model_config['num_hidden_layers']}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Data Type: {args.dtype} ({dtype_size} bytes)")
    print(f"TP Size: {args.tp_size}")
    print(f"PP Size: {args.pp_size}")
    print(f"Micro-Batch Size: {args.micro_batch_size}")
    
    # Calculate TP communication volume
    tp_comm = calculate_tp_communication_volume(
        hidden_size=model_config['hidden_size'],
        intermediate_size=model_config['intermediate_size'],
        num_attention_heads=model_config['num_attention_heads'],
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        tp_size=args.tp_size,
        dtype_size=dtype_size,
    )
    
    # Calculate PP communication volume
    pp_comm = calculate_pipeline_communication_volume(
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_hidden_layers'],
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        pp_size=args.pp_size,
        dtype_size=dtype_size,
    )
    
    # Print results
    print("\n--- Tensor Parallelism Communication ---")
    print(f"Forward Pass: {tp_comm['forward_gb']:.3f} GB")
    print(f"Backward Pass: {tp_comm['backward_gb']:.3f} GB")
    print(f"Total TP Communication: {tp_comm['total_gb']:.3f} GB")
    
    print("\n--- Pipeline Parallelism Communication ---")
    print(f"Forward Pass: {pp_comm['forward_gb']:.3f} GB")
    print(f"Backward Pass: {pp_comm['backward_gb']:.3f} GB")
    print(f"Total PP Communication: {pp_comm['total_gb']:.3f} GB")
    
    total_comm_gb = tp_comm['total_gb'] + pp_comm['total_gb']
    print(f"\nTotal Communication (TP+PP): {total_comm_gb:.3f} GB")


if __name__ == "__main__":
    main()
