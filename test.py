#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

def plot(events, N, title):
    kind_color = {'F':'skyblue', 'B':'salmon', 'Csend':'lightgreen','Crecv':'palegreen'}
    fig, ax = plt.subplots(figsize=(16,8))
    for g,t0,t1,k,mb in events:
        y = N - g - 0.8
        h = 0.6 if k in ('F','B') else 0.3
        rect = Rectangle((t0,y), t1-t0, h, color=kind_color[k])
        ax.add_patch(rect)
        if k in ('F','B'):
            ax.text(t0+(t1-t0)/2, y+0.3, f'{k}{mb}', ha='center', va='center', fontsize=9)
    t_end = max(e[2] for e in events)
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
    plt.tight_layout();  plt.show()
    return t_end

if __name__ == '__main__':
    M, N, F, B, C = 8, 4, 10.0, 20.0, 2.0
    events = schedule_1f1b(M, N, F, B, C)
    T = plot(events, N,
             f'1F1B Pipeline (M={M}, N={N}, F={F} ms, B={B} ms, C={C} ms)')
    print(f'End-to-End   : {T:.1f} ms')
    print(f'Theoretical  : {(F+B)+(N-1)*max(F,B)+2*C:.1f} ms')
