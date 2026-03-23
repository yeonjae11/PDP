#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.patches import Rectangle

def visualize_1f1b(num_microbatches, forward_time, backward_time, comm_time, output_file=None):
    """
    Visualize the 1F1B pipeline parallelism pattern across 4 GPUs.
    
    Args:
        num_microbatches: Number of microbatches in the pipeline
        forward_time: Time (ms) for forward pass per microbatch
        backward_time: Time (ms) for backward pass per microbatch
        comm_time: Time (ms) for communication between GPUs
        output_file: If provided, save figure to this file
    """
    num_gpus = 4
    
    # Colors for different operations
    colors = {
        'idle': 'white',
        'forward': 'skyblue',
        'backward': 'salmon',
        'comm_send': 'lightgreen',
        'comm_recv': 'palegreen'
    }
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Track the current time for each GPU
    gpu_times = [0] * num_gpus
    max_time = 0
    
    # Track all blocks for legend creation
    forward_block = None
    backward_block = None
    comm_send_block = None
    comm_recv_block = None
    
    # Dictionary to store all operations for analysis
    operations = []
    
    # Schedule micro-batches according to 1F1B
    active_microbatches = []
    
    # Initial forward passes
    for mb in range(min(num_gpus, num_microbatches)):
        for gpu in range(mb + 1):
            if gpu < num_gpus:
                # Add forward pass
                start_time = gpu_times[gpu]
                end_time = start_time + forward_time
                forward_block = Rectangle((start_time, num_gpus - gpu - 0.8), forward_time, 0.6, 
                                        color=colors['forward'], label='Forward')
                ax.add_patch(forward_block)
                ax.text(start_time + forward_time/2, num_gpus - gpu - 0.5, f'F{mb}', 
                        ha='center', va='center', fontsize=9)
                
                # Add to operations list
                operations.append({
                    'gpu': gpu,
                    'type': 'forward',
                    'microbatch': mb,
                    'start': start_time,
                    'end': end_time
                })
                
                gpu_times[gpu] = end_time
                max_time = max(max_time, end_time)
                
                # Add communication time if not the last GPU
                if gpu < num_gpus - 1:
                    start_time = gpu_times[gpu]
                    end_time = start_time + comm_time
                    comm_send_block = Rectangle((start_time, num_gpus - gpu - 0.8), comm_time, 0.3, 
                                             color=colors['comm_send'], label='Comm Send')
                    ax.add_patch(comm_send_block)
                    
                    # Add corresponding receive on next GPU
                    comm_recv_block = Rectangle((start_time, num_gpus - gpu - 1 - 0.5), comm_time, 0.3, 
                                             color=colors['comm_recv'], label='Comm Receive')
                    ax.add_patch(comm_recv_block)
                    
                    # Add to operations list
                    operations.append({
                        'gpu': gpu,
                        'type': 'comm_send',
                        'microbatch': mb,
                        'start': start_time,
                        'end': end_time
                    })
                    operations.append({
                        'gpu': gpu + 1,
                        'type': 'comm_recv',
                        'microbatch': mb,
                        'start': start_time,
                        'end': end_time
                    })
                    
                    gpu_times[gpu] = end_time
                    gpu_times[gpu + 1] = max(gpu_times[gpu + 1], end_time)
                    max_time = max(max_time, end_time)
    
    # 1F1B steady state
    current_mb_forward = num_gpus
    current_mb_backward = 0
    
    while current_mb_forward < num_microbatches or current_mb_backward < num_microbatches:
        # Process a backward pass if possible
        if current_mb_backward < min(current_mb_forward, num_microbatches):
            for gpu in range(num_gpus - 1, -1, -1):
                # Add backward pass
                start_time = gpu_times[gpu]
                end_time = start_time + backward_time
                backward_block = Rectangle((start_time, num_gpus - gpu - 0.8), backward_time, 0.6, 
                                         color=colors['backward'], label='Backward')
                ax.add_patch(backward_block)
                ax.text(start_time + backward_time/2, num_gpus - gpu - 0.5, f'B{current_mb_backward}', 
                        ha='center', va='center', fontsize=9)
                
                # Add to operations list
                operations.append({
                    'gpu': gpu,
                    'type': 'backward',
                    'microbatch': current_mb_backward,
                    'start': start_time,
                    'end': end_time
                })
                
                gpu_times[gpu] = end_time
                max_time = max(max_time, end_time)
                
                # Add communication time if not the first GPU
                if gpu > 0:
                    start_time = gpu_times[gpu]
                    end_time = start_time + comm_time
                    comm_send_block = Rectangle((start_time, num_gpus - gpu - 0.8), comm_time, 0.3, 
                                             color=colors['comm_send'], label='Comm Send')
                    ax.add_patch(comm_send_block)
                    
                    # Add corresponding receive on previous GPU
                    comm_recv_block = Rectangle((start_time, num_gpus - gpu + 1 - 0.5), comm_time, 0.3, 
                                             color=colors['comm_recv'], label='Comm Receive')
                    ax.add_patch(comm_recv_block)
                    
                    # Add to operations list
                    operations.append({
                        'gpu': gpu,
                        'type': 'comm_send',
                        'microbatch': current_mb_backward,
                        'start': start_time,
                        'end': end_time
                    })
                    operations.append({
                        'gpu': gpu - 1,
                        'type': 'comm_recv',
                        'microbatch': current_mb_backward,
                        'start': start_time,
                        'end': end_time
                    })
                    
                    gpu_times[gpu] = end_time
                    gpu_times[gpu - 1] = max(gpu_times[gpu - 1], end_time)
                    max_time = max(max_time, end_time)
            
            current_mb_backward += 1
        
        # Process a forward pass if possible
        if current_mb_forward < num_microbatches:
            for gpu in range(num_gpus):
                # Add forward pass
                start_time = gpu_times[gpu]
                end_time = start_time + forward_time
                forward_block = Rectangle((start_time, num_gpus - gpu - 0.8), forward_time, 0.6, 
                                        color=colors['forward'], label='Forward')
                ax.add_patch(forward_block)
                ax.text(start_time + forward_time/2, num_gpus - gpu - 0.5, f'F{current_mb_forward}', 
                        ha='center', va='center', fontsize=9)
                
                # Add to operations list
                operations.append({
                    'gpu': gpu,
                    'type': 'forward',
                    'microbatch': current_mb_forward,
                    'start': start_time,
                    'end': end_time
                })
                
                gpu_times[gpu] = end_time
                max_time = max(max_time, end_time)
                
                # Add communication time if not the last GPU
                if gpu < num_gpus - 1:
                    start_time = gpu_times[gpu]
                    end_time = start_time + comm_time
                    comm_send_block = Rectangle((start_time, num_gpus - gpu - 0.8), comm_time, 0.3, 
                                             color=colors['comm_send'], label='Comm Send')
                    ax.add_patch(comm_send_block)
                    
                    # Add corresponding receive on next GPU
                    comm_recv_block = Rectangle((start_time, num_gpus - gpu - 1 - 0.5), comm_time, 0.3, 
                                             color=colors['comm_recv'], label='Comm Receive')
                    ax.add_patch(comm_recv_block)
                    
                    # Add to operations list
                    operations.append({
                        'gpu': gpu,
                        'type': 'comm_send',
                        'microbatch': current_mb_forward,
                        'start': start_time,
                        'end': end_time
                    })
                    operations.append({
                        'gpu': gpu + 1,
                        'type': 'comm_recv',
                        'microbatch': current_mb_forward,
                        'start': start_time,
                        'end': end_time
                    })
                    
                    gpu_times[gpu] = end_time
                    gpu_times[gpu + 1] = max(gpu_times[gpu + 1], end_time)
                    max_time = max(max_time, end_time)
            
            current_mb_forward += 1
    
    # Remaining backward passes
    while current_mb_backward < num_microbatches:
        for gpu in range(num_gpus - 1, -1, -1):
            # Add backward pass
            start_time = gpu_times[gpu]
            end_time = start_time + backward_time
            backward_block = Rectangle((start_time, num_gpus - gpu - 0.8), backward_time, 0.6, 
                                     color=colors['backward'], label='Backward')
            ax.add_patch(backward_block)
            ax.text(start_time + backward_time/2, num_gpus - gpu - 0.5, f'B{current_mb_backward}', 
                    ha='center', va='center', fontsize=9)
            
            # Add to operations list
            operations.append({
                'gpu': gpu,
                'type': 'backward',
                'microbatch': current_mb_backward,
                'start': start_time,
                'end': end_time
            })
            
            gpu_times[gpu] = end_time
            max_time = max(max_time, end_time)
            
            # Add communication time if not the first GPU
            if gpu > 0:
                start_time = gpu_times[gpu]
                end_time = start_time + comm_time
                comm_send_block = Rectangle((start_time, num_gpus - gpu - 0.8), comm_time, 0.3, 
                                         color=colors['comm_send'], label='Comm Send')
                ax.add_patch(comm_send_block)
                
                # Add corresponding receive on previous GPU
                comm_recv_block = Rectangle((start_time, num_gpus - gpu + 1 - 0.5), comm_time, 0.3, 
                                         color=colors['comm_recv'], label='Comm Receive')
                ax.add_patch(comm_recv_block)
                
                # Add to operations list
                operations.append({
                    'gpu': gpu,
                    'type': 'comm_send',
                    'microbatch': current_mb_backward,
                    'start': start_time,
                    'end': end_time
                })
                operations.append({
                    'gpu': gpu - 1,
                    'type': 'comm_recv',
                    'microbatch': current_mb_backward,
                    'start': start_time,
                    'end': end_time
                })
                
                gpu_times[gpu] = end_time
                gpu_times[gpu - 1] = max(gpu_times[gpu - 1], end_time)
                max_time = max(max_time, end_time)
        
        current_mb_backward += 1
    
    # Calculate end-to-end time
    end_to_end_time = max(gpu_times)
    
    # Set up the axes
    ax.set_xlim(0, end_to_end_time * 1.05)
    ax.set_ylim(0, num_gpus)
    
    # Set labels and title
    ax.set_xlabel('Time (ms)')
    y_labels = [f'GPU {i}' for i in range(num_gpus)]
    ax.set_yticks([num_gpus - i - 0.5 for i in range(num_gpus)])
    ax.set_yticklabels(y_labels)
    
    title = f'1F1B Pipeline Parallelism - {num_microbatches} Microbatches\n'
    title += f'Forward: {forward_time}ms, Backward: {backward_time}ms, Comm: {comm_time}ms\n'
    title += f'End-to-End Time: {end_to_end_time:.2f}ms'
    ax.set_title(title)
    
    # Create custom legend
    legend_handles = [
        Rectangle((0, 0), 1, 1, color=colors['forward'], label='Forward'),
        Rectangle((0, 0), 1, 1, color=colors['backward'], label='Backward'),
        Rectangle((0, 0), 1, 1, color=colors['comm_send'], label='Communication')
    ]
    ax.legend(handles=legend_handles, loc='upper right')
    
    # Draw grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Display end-to-end time
    ax.axvline(x=end_to_end_time, color='red', linestyle='--', linewidth=2)
    ax.text(end_to_end_time, num_gpus * 0.5, f' End-to-End: {end_to_end_time:.2f}ms', 
            color='red', fontsize=12, va='center')
    
    # Save the figure if output file is provided
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Pipeline Configuration:")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Number of Microbatches: {num_microbatches}")
    print(f"  Forward Pass Time: {forward_time}ms")
    print(f"  Backward Pass Time: {backward_time}ms")
    print(f"  Communication Time: {comm_time}ms")
    print(f"  End-to-End Time: {end_to_end_time:.2f}ms")
    
    # Calculate theoretical throughput
    samples_per_iteration = num_microbatches  # Assuming 1 sample per microbatch
    throughput = samples_per_iteration / (end_to_end_time / 1000)  # samples per second
    print(f"  Theoretical Throughput: {throughput:.2f} samples/second")
    
    return end_to_end_time, operations

def calculate_efficiency(operations, end_to_end_time, num_gpus=4):
    """
    Calculate pipeline efficiency based on GPU utilization.
    """
    # Calculate total compute time across all GPUs
    total_compute_time = sum(
        op['end'] - op['start'] 
        for op in operations 
        if op['type'] in ['forward', 'backward']
    )
    
    # Maximum possible compute time
    max_compute_time = end_to_end_time * num_gpus
    
    # Efficiency is the ratio of actual compute to maximum possible compute
    efficiency = (total_compute_time / max_compute_time) * 100
    
    print(f"  Pipeline Efficiency: {efficiency:.2f}%")
    return efficiency

def compare_configurations(configs, output_file=None):
    """
    Compare different pipeline configurations.
    
    Args:
        configs: List of tuples (num_microbatches, forward_time, backward_time, comm_time, label)
        output_file: If provided, save figure to this file
    """
    results = []
    
    for config in configs:
        num_microbatches, forward_time, backward_time, comm_time, label = config
        end_to_end_time, operations = visualize_1f1b(
            num_microbatches, forward_time, backward_time, comm_time
        )
        efficiency = calculate_efficiency(operations, end_to_end_time)
        results.append((label, end_to_end_time, efficiency))
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    labels = [r[0] for r in results]
    times = [r[1] for r in results]
    efficiencies = [r[2] for r in results]
    
    # Plot end-to-end times
    ax1.bar(labels, times, color='skyblue')
    ax1.set_ylabel('End-to-End Time (ms)')
    ax1.set_title('End-to-End Training Time Comparison')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot efficiencies
    ax2.bar(labels, efficiencies, color='lightgreen')
    ax2.set_ylabel('Pipeline Efficiency (%)')
    ax2.set_title('Pipeline Efficiency Comparison')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Comparison saved to {output_file}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize 1F1B Pipeline Parallelism')
    parser.add_argument('--microbatches', type=int, default=8, 
                        help='Number of microbatches')
    parser.add_argument('--forward', type=float, default=10.0, 
                        help='Forward pass time (ms)')
    parser.add_argument('--backward', type=float, default=20.0, 
                        help='Backward pass time (ms)')
    parser.add_argument('--comm', type=float, default=2.0, 
                        help='Communication time (ms)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file to save the visualization')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different configurations')
    
    args = parser.parse_args()
    
    if args.compare:
        # Example comparison of different configurations
        configs = [
            (4, args.forward, args.backward, args.comm, "4 MBs"),
            (8, args.forward, args.backward, args.comm, "8 MBs"),
            (16, args.forward, args.backward, args.comm, "16 MBs"),
        ]
        compare_configurations(configs, args.output)
    else:
        end_to_end_time, operations = visualize_1f1b(
            args.microbatches, args.forward, args.backward, args.comm, args.output
        )
        calculate_efficiency(operations, end_to_end_time)
