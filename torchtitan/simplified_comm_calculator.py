import argparse

def calculate_tp_communication_single_block(
    hidden_size: int,
    intermediate_size: int,
    seq_length: int,
    batch_size: int,
    tp_size: int,
    dtype_size: int = 2  # Default to BF16 (2 bytes)
):
    """
    Calculate communication volume for tensor parallelism in a single Llama transformer block.
    
    Args:
        hidden_size: Model hidden dimension size
        intermediate_size: Size of intermediate feed-forward dimension
        seq_length: Sequence length
        batch_size: Batch size
        tp_size: Tensor parallel size (number of GPUs)
        dtype_size: Size of data type in bytes (default: 2 for BF16)
        
    Returns:
        Dictionary with communication volume breakdown
    """
    # Size per sample (B×S = batch_size × seq_length)
    if tp_size == 1:
        return {
            "forward_reduce_scatter_mb": 0,
            "forward_all_gather_mb": 0,
            "backward_reduce_scatter_mb": 0,
            "backward_all_gather_mb": 0,
        }
    bs_size = batch_size * seq_length
    
    comm_volume = {}
    
    tensor_size_mb = bs_size * hidden_size * dtype_size / (1024 * 1024)
    comm_volume["forward_reduce_scatter_mb"] = tensor_size_mb
    
    comm_volume["forward_all_gather_mb"] = tensor_size_mb
    
    backward_rs_mb = tensor_size_mb * 3
    comm_volume["backward_reduce_scatter_mb"] = backward_rs_mb
    
    backward_ag_mb = tensor_size_mb * 4
    comm_volume["backward_all_gather_mb"] = backward_ag_mb
    
    for key, value in comm_volume.items():
        comm_volume[key] = value * (tp_size-1)/tp_size 
    
    return comm_volume


def get_pipeline_block_sizes(
    hidden_size: int,
    seq_length: int,
    batch_size: int,
    tp_size: int,
    dtype_size: int = 2  # Default to BF16 (2 bytes)
):
    """
    Calculate input and gradient sizes for pipeline parallelism with tensor parallelism.
    
    Args:
        hidden_size: Model hidden dimension size
        seq_length: Sequence length
        batch_size: Micro-batch size for pipeline parallelism
        tp_size: Tensor parallel size (number of GPUs)
        dtype_size: Size of data type in bytes (default: 2 for BF16)
        
    Returns:
        Activation size in MB for pipeline communication
    """
    activation_size_mb = batch_size * (seq_length // tp_size) * hidden_size * dtype_size / (1024 * 1024)
    
    return activation_size_mb


def main():
    parser = argparse.ArgumentParser(description="Calculate communication volume for parallel training")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden size")
    parser.add_argument("--intermediate_size", type=int, default=14336, help="Intermediate size in FFN")
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor parallel size")
    
    args = parser.parse_args()
    
    # Fixed to bfloat16 (2 bytes)
    dtype_size = 2
    
    print(f"\n--- Communication Analysis for a Single Transformer Block ---")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"Intermediate Size: {args.intermediate_size}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"TP Size: {args.tp_size}")
    print(f"Data Type: bfloat16 ({dtype_size} bytes)")
    
    # Calculate TP communication volume
    tp_comm = calculate_tp_communication_single_block(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        tp_size=args.tp_size,
        dtype_size=dtype_size,
    )
    
    # Get pipeline sizes
    pp_sizes = get_pipeline_block_sizes(
        hidden_size=args.hidden_size,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        tp_size=args.tp_size,
        dtype_size=dtype_size,
    )
    
    # Print results
    print("\n--- Tensor Parallelism Communication (Single Block) ---")
    print(f"Forward Pass:")
    print(f"  Reduce-Scatter: {tp_comm['forward_reduce_scatter_mb']:.2f} MB")
    print(f"  All-Gather: {tp_comm['forward_all_gather_mb']:.2f} MB")
    print(f"Backward Pass:")
    print(f"  Reduce-Scatter (3 operations): {tp_comm['backward_reduce_scatter_mb']:.2f} MB")
    print(f"  All-Gather (4 operations): {tp_comm['backward_all_gather_mb']:.2f} MB")
    
    print("\n--- Pipeline Communication Sizes (with TP) ---")
    print(f"Forward Activation Size: {pp_sizes:.2f} MB")

if __name__ == "__main__":
    main()
