#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse

def schedule_1f1b(
        M=8,                # micro-batches
        N=4,                # pipeline stages = GPUs
        F=10.0,             # forward(ms)
        B=20.0,             # backward(ms)
        C=2.0,              # comm(ms)
):
    gpu_free  = [0.0]*N          # 각 GPU가 비는 시간
    done_fwd  = [[-1.0]*N for _ in range(M)]
    done_bwd  = [[-1.0]*N for _ in range(M)]
    events    = []               # (gpu, t0, t1, kind, mb)
    
    # 총 step: fill+steady+drain = 2M+N-2
    for step in range(2*M + N - 2):
        # --- 선택할 마이크로배치 & 스테이지 ----------------------------
        if step < N:                       # 채우기 구간
            mb_fwd = step
            stg_f  = 0
        else:                              # 스테디 구간
            mb_fwd = step - (N-1)
            stg_f  = (step) % N
        mb_bwd = step - (N-1)
        stg_b  = N-1 - ((step) % N)
        
        # --- Forward ----------------------------------------------------
        if 0 <= mb_fwd < M:
            s = stg_f
            mb = mb_fwd
            # 의존성: (1) GPU가 놀아야 하고 (2) 앞 스테이지 결과를 받아야 함
            start = max(
                gpu_free[s],
                0 if s==0 else done_fwd[mb][s-1] + C
            )
            end = start + F
            done_fwd[mb][s] = end
            gpu_free[s]     = end
            events.append((s, start, end, 'F', mb))
            if s < N-1:  # 통신(Send/Recv)
                events.append((s,     end, end+C, 'Csend', mb))
                events.append((s+1,   end, end+C, 'Crecv', mb))
        
        # --- Backward ---------------------------------------------------
        if 0 <= mb_bwd < M:
            s = stg_b
            mb = mb_bwd
            # 아직 forward 끝나지 않았으면 skip
            if done_fwd[mb][s] >= 0:
                start = max(
                    gpu_free[s],
                    0 if s==N-1 else done_bwd[mb][s+1] + C,
                    done_fwd[mb][s]          # 반드시 forward 이후
                )
                end = start + B
                done_bwd[mb][s] = end
                gpu_free[s]     = end
                events.append((s, start, end, 'B', mb))
                if s > 0:
                    events.append((s,   end, end+C, 'Csend', mb))
                    events.append((s-1, end, end+C, 'Crecv', mb))
    return events

def plot(events, N, title, output_file=None):
    kind_color = {'F':'skyblue', 'B':'salmon', 'Csend':'lightgreen','Crecv':'palegreen'}
    fig, ax = plt.subplots(figsize=(16,8))
    
    # Draw all events
    for g,t0,t1,k,mb in events:
        y = N - g - 0.8
        h = 0.6 if k in ('F','B') else 0.3
        rect = Rectangle((t0,y), t1-t0, h, color=kind_color[k])
        ax.add_patch(rect)
        if k in ('F','B'):
            ax.text(t0+(t1-t0)/2, y+0.3, f'{k}{mb}', ha='center', va='center', fontsize=9)
    
    # Calculate end-to-end time
    t_end = max(e[2] for e in events)
    
    # Set up the plot
    ax.set_xlim(0, t_end*1.05)
    ax.set_ylim(0, N)
    ax.set_yticks([N-i-0.5 for i in range(N)])
    ax.set_yticklabels([f'GPU {i}' for i in range(N)])
    ax.set_xlabel('Time (ms)')
    ax.set_title(title + f'\nEnd-to-End = {t_end:.1f} ms')
    ax.axvline(t_end, color='red', ls='--')
    ax.legend([Rectangle((0,0),1,1,color=c) for c in ('skyblue','salmon','lightgreen')],
              ['Forward','Backward','Comm'], loc='upper right')
    ax.grid(True, ls='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")
    
    plt.show()
    return t_end

def calculate_efficiency(events, t_end, N):
    """Calculate pipeline efficiency based on GPU utilization"""
    # Sum up compute time (F/B operations only, not communication)
    compute_time = sum(
        e[2] - e[1] for e in events 
        if e[3] in ('F', 'B')
    )
    
    # Maximum possible compute time
    max_compute_time = t_end * N
    
    # Calculate efficiency
    efficiency = (compute_time / max_compute_time) * 100
    
    print(f"Pipeline Efficiency: {efficiency:.2f}%")
    return efficiency

def compare_configs(configs, output_file=None):
    """Compare different pipeline configurations"""
    results = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for config in configs:
        M, N, F, B, C, label = config
        events = schedule_1f1b(M, N, F, B, C)
        t_end = max(e[2] for e in events)
        
        # Calculate efficiency
        compute_time = sum(e[2] - e[1] for e in events if e[3] in ('F', 'B'))
        max_compute_time = t_end * N
        efficiency = (compute_time / max_compute_time) * 100
        
        results.append((label, t_end, efficiency))
    
    # Plot results
    labels = [r[0] for r in results]
    times = [r[1] for r in results]
    efficiencies = [r[2] for r in results]
    
    ax1.bar(labels, times, color='skyblue')
    ax1.set_ylabel('End-to-End Time (ms)')
    ax1.set_title('End-to-End Training Time Comparison')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.bar(labels, efficiencies, color='lightgreen')
    ax2.set_ylabel('Pipeline Efficiency (%)')
    ax2.set_title('Pipeline Efficiency Comparison')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Comparison saved to {output_file}")
    
    plt.show()
    
    # Print results
    print("\nComparison Results:")
    print("-" * 50)
    print(f"{'Configuration':<15} {'End-to-End (ms)':<20} {'Efficiency (%)':<15}")
    print("-" * 50)
    for label, t_end, efficiency in results:
        print(f"{label:<15} {t_end:<20.2f} {efficiency:<15.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 1F1B Pipeline Parallelism')
    parser.add_argument('--microbatches', '-m', type=int, default=8, 
                        help='Number of microbatches')
    parser.add_argument('--gpus', '-g', type=int, default=4, 
                        help='Number of GPUs (pipeline stages)')
    parser.add_argument('--forward', '-f', type=float, default=10.0, 
                        help='Forward pass time (ms)')
    parser.add_argument('--backward', '-b', type=float, default=20.0, 
                        help='Backward pass time (ms)')
    parser.add_argument('--comm', '-c', type=float, default=2.0, 
                        help='Communication time (ms)')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output file to save visualization')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different configurations')
    
    args = parser.parse_args()
    
    if args.compare:
        # Example of comparing different configurations
        configs = [
            (4, args.gpus, args.forward, args.backward, args.comm, "4 MBs"),
            (8, args.gpus, args.forward, args.backward, args.comm, "8 MBs"),
            (16, args.gpus, args.forward, args.backward, args.comm, "16 MBs"),
        ]
        compare_configs(configs, args.output)
    else:
        events = schedule_1f1b(args.microbatches, args.gpus, 
                              args.forward, args.backward, args.comm)
        t_end = plot(events, args.gpus, 
                    f'1F1B Pipeline (M={args.microbatches}, N={args.gpus}, F={args.forward} ms, B={args.backward} ms, C={args.comm} ms)',
                    args.output)
        
        # Calculate theoretical time
        theoretical = (args.forward + args.backward) + (args.gpus-1) * max(args.forward, args.backward) + 2*args.comm
        print(f'End-to-End   : {t_end:.1f} ms')
        print(f'Theoretical  : {theoretical:.1f} ms')
        
        # Calculate efficiency
        calculate_efficiency(events, t_end, args.gpus)
