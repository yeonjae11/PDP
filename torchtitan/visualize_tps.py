import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


def get_strategy_display_name(strategy):
    """
    Get display name for strategy with clear, readable formatting.
    """
    if strategy == '_1f1b':
        return '1F1B'
    elif strategy == '_I1f1b':
        return 'I-1F1B'
    elif strategy == '_tp':
        return 'TP'
    elif strategy == '_tp_1f1b':
        return 'TP-1F1B'
    elif strategy == '_tp_I1f1b':
        return 'TP-I-1F1B'
    elif strategy == '_zero':
        return 'FSDP'
    else:
        return strategy.strip('_')

# Set nicer plotting style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.unicode_minus'] = False

def load_tps_data(log_dir, include_zero=True):
    """
    Load TPS data from specified directory containing model subdirectories.
    Each model directory should contain a text file with the same name containing TPS data.
    
    Args:
        log_dir (str): Directory containing model subdirectories
        include_zero (bool): Whether to include the _zero (FSDP) strategy data
    
    Returns:
        pd.DataFrame: DataFrame with columns [model, sequence_length, batch_size, strategy, tps]
    """
    all_data = []
    
    model_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    print(f"Found model directories: {model_dirs}")
    
    for model_dir in model_dirs:
        model_path = os.path.join(log_dir, model_dir)
        data_file = os.path.join(model_path, f"{model_dir}.txt")
        
        if not os.path.exists(data_file):
            print(f"Warning: No data file found for {model_dir}")
            continue
            
        print(f"Loading data from {data_file}")
        
        with open(data_file, 'r') as f:
            lines = f.readlines()
            print(f"Read {len(lines)} lines from {model_dir}.txt")
            
            for line in lines:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                    
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                    
                config, tps_value = parts
                config = config.strip()
                
                try:
                    tps_value = float(tps_value.strip())
                except ValueError:
                    continue
                
                # Skip zero or very small TPS values
                if tps_value <= 0.01:
                    continue
                
                # Extract strategy information
                strategy = None
                possible_strategies = ['_tp_1f1b', '_tp_I1f1b', '_tp', '_1f1b', '_I1f1b', '_zero']
                # 길이 순으로 정렬하여 더 긴(더 구체적인) 전략을 먼저 검색
                possible_strategies = sorted(possible_strategies, key=len, reverse=True)
                
                for s in possible_strategies:
                    if s in config:
                        strategy = s
                        break
                
                if strategy is None:
                    print(f"Warning: Could not determine strategy for config: {config}")
                    continue
                    
                # Skip _zero strategy if not included
                if not include_zero and strategy == '_zero':
                    continue
                
                # Extract sequence length
                sl_match = re.search(r'sl(\d+)', config)
                if sl_match:
                    sequence_length = int(sl_match.group(1))
                else:
                    # Default to sequence length 1 if not specified
                    sequence_length = 1
                
                # Extract batch size
                bs_match = re.search(r'bs(\d+)', config)
                if bs_match:
                    batch_size = int(bs_match.group(1))
                else:
                    # Default to batch size 16 if not specified
                    batch_size = 16
                
                # Add the data point
                all_data.append({
                    'model': model_dir,
                    'strategy': strategy,
                    'batch_size': batch_size,
                    'sequence_length': sequence_length,
                    'tps': tps_value
                })
    
    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        print(f"Loaded data summary:")
        print(f"Models: {df['model'].unique()}")
        print(f"Strategies: {df['strategy'].unique()}")
        print(f"Total entries: {len(df)}")
        return df
    else:
        print("Warning: No data was loaded!")
        return pd.DataFrame(columns=['model', 'strategy', 'batch_size', 'sequence_length', 'tps'])

def plot_by_model_strategy(df, output_dir=None, figsize=(14, 8)):
    """
    Create a grouped bar plot comparing TPS by model and parallelism strategy.
    
    Args:
        df (pd.DataFrame): DataFrame with TPS data
        output_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Clean strategy names for better display
    df_plot = df.copy()
    df_plot['strategy_display'] = df_plot['strategy'].apply(get_strategy_display_name)
    
    # Create a pivot table with models as rows and strategies as columns
    pivot_data = df_plot.pivot_table(
        values='tps', 
        index='model', 
        columns='strategy_display', 
        aggfunc='mean'
    )
    
    # Plot the data
    ax = pivot_data.plot(kind='bar', figsize=figsize)
    
    plt.title('TPS by Model and Parallelism Strategy', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Tokens Per Second (TPS)', fontsize=14)
    plt.xticks(rotation=25)
    
    # Fix legend display
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Strategy', fontsize=12, title_fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add TPS values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'tps_by_model_strategy.png'), dpi=300)
        
    plt.show()

def plot_by_sequence_length(df, output_dir=None, figsize=(14, 8)):
    """
    Create a grouped bar plot comparing TPS by sequence length across models and strategies.
    
    Args:
        df (pd.DataFrame): DataFrame with TPS data
        output_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size (width, height)
    """
    # Clean strategy names for better display
    df_plot = df.copy()
    df_plot['strategy_display'] = df_plot['strategy'].apply(get_strategy_display_name)
    
    # Group by model, strategy, and sequence_length
    grouped_df = df_plot.groupby(['model', 'strategy_display', 'sequence_length']).agg({
        'tps': 'mean'
    }).reset_index()
    
    # Plot the data
    g = sns.catplot(
        data=grouped_df,
        x='sequence_length',
        y='tps',
        hue='strategy_display',
        col='model',
        kind='bar',
        height=6,
        aspect=0.8,
        sharey=True,
        legend_out=True
    )
    
    g.fig.suptitle('TPS by Sequence Length', fontsize=16, y=1.02)
    g.set_axis_labels('Sequence Length', 'Tokens Per Second (TPS)')
    g.set_titles("{col_name}", fontsize=14)
    
    # Add value labels on bars
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', 
                        (p.get_x() + p.get_width()/2., p.get_height()), 
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Improve the legend
    plt.subplots_adjust(right=0.85)
    handles, labels = g.axes[0,0].get_legend_handles_labels()
    g.fig.legend(handles, labels, title='Strategy', bbox_to_anchor=(0.95, 0.5), loc='center right', fontsize=10)
    
    # Adjust layout
    g.fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'tps_by_sequence_length.png'), dpi=300)
        
    plt.show()

def plot_by_batch_size(df, output_dir=None, figsize=(14, 8)):
    """
    Create a grouped bar plot comparing TPS by batch size across models and strategies.
    
    Args:
        df (pd.DataFrame): DataFrame with TPS data
        output_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size (width, height)
    """
    # Clean strategy names for better display
    df_plot = df.copy()
    df_plot['strategy_display'] = df_plot['strategy'].apply(get_strategy_display_name)
    
    # Group by model, strategy, and batch_size
    grouped_df = df_plot.groupby(['model', 'strategy_display', 'batch_size']).agg({
        'tps': 'mean'
    }).reset_index()
    
    # Plot the data
    g = sns.catplot(
        data=grouped_df,
        x='batch_size',
        y='tps',
        hue='strategy_display',
        col='model',
        kind='bar',
        height=6,
        aspect=0.8,
        sharey=True,
        legend_out=True
    )
    
    g.fig.suptitle('TPS by Batch Size', fontsize=16, y=1.02)
    g.set_axis_labels('Batch Size', 'Tokens Per Second (TPS)')
    g.set_titles("{col_name}", fontsize=14)
    
    # Add value labels on bars
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', 
                        (p.get_x() + p.get_width()/2., p.get_height()), 
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Improve the legend
    plt.subplots_adjust(right=0.85)
    handles, labels = g.axes[0,0].get_legend_handles_labels()
    g.fig.legend(handles, labels, title='Strategy', bbox_to_anchor=(0.95, 0.5), loc='center right', fontsize=10)
    
    # Adjust layout
    g.fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'tps_by_batch_size.png'), dpi=300)
        
    plt.show()

def plot_heatmap(df, output_dir=None, figsize=(16, 12)):
    """
    Create a heatmap comparison of TPS values across configurations.
    
    Args:
        df (pd.DataFrame): DataFrame with TPS data
        output_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size (width, height)
    """
    # Clean strategy names for better display
    df_plot = df.copy()
    df_plot['strategy_display'] = df_plot['strategy'].apply(get_strategy_display_name)
    
    # Create a figure with subplots for each model
    models = df_plot['model'].unique()
    fig, axes = plt.subplots(len(models), 1, figsize=figsize)
    
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        model_data = df_plot[df_plot['model'] == model]
        
        # Create a pivot table for the heatmap
        pivot = model_data.pivot_table(
            values='tps',
            index='strategy_display',
            columns=['sequence_length', 'batch_size'],
            aggfunc='mean'
        )
        
        # Plot the heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=axes[i], 
                    cbar=(i == len(models) - 1), linewidths=0.5, annot_kws={'fontsize': 9})
        axes[i].set_title(f'TPS Heatmap for {model}', fontsize=14)
        axes[i].set_ylabel('Strategy', fontsize=12)
        
        # Format x-axis labels
        if i == len(models) - 1:  # Only for the last subplot
            axes[i].set_xlabel('(Sequence Length, Batch Size)', fontsize=12)
        
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'tps_heatmap.png'), dpi=300)
        
    plt.show()
    
def plot_comprehensive_comparison(df, output_dir=None, figsize=(18, 14)):
    """
    Create a comprehensive visualization showing the relationship between
    sequence length, batch size, and TPS across all models and strategies.
    
    Args:
        df (pd.DataFrame): DataFrame with TPS data
        output_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size (width, height)
    """
    # Clean strategy names for better display
    df_plot = df.copy()
    df_plot['strategy_display'] = df_plot['strategy'].apply(get_strategy_display_name)
    
    # Create a figure with subplots for different sequence lengths
    seq_lengths = sorted(df_plot['sequence_length'].unique())
    batch_sizes = sorted(df_plot['batch_size'].unique())
    models = sorted(df_plot['model'].unique())
    
    fig, axes = plt.subplots(len(seq_lengths), 1, figsize=figsize, sharex=True)
    
    if len(seq_lengths) == 1:
        axes = [axes]
    
    # Set up colors for strategies
    strategies = sorted(df_plot['strategy_display'].unique())
    colors = sns.color_palette("husl", len(strategies))
    strategy_colors = dict(zip(strategies, colors))
    
    # Create subplots for each sequence length
    for i, sl in enumerate(seq_lengths):
        ax = axes[i]
        sl_data = df_plot[df_plot['sequence_length'] == sl]
        
        # Plot grouped bars for each batch size
        width = 0.15  # Width of each bar
        offset_multiplier = np.arange(len(strategies)) - len(strategies)/2 + 0.5
        
        # Set up positions for x-axis (batch sizes)
        x_positions = np.arange(len(models))
        
        # Plot each strategy as a group at each batch size
        for j, strat in enumerate(strategies):
            strat_data = sl_data[sl_data['strategy_display'] == strat]
            
            # Group by model and calculate mean TPS
            model_tps = []
            for model in models:
                model_data = strat_data[strat_data['model'] == model]
                if not model_data.empty:
                    model_tps.append(model_data['tps'].mean())
                else:
                    model_tps.append(0)  # No data for this combination
            
            # Skip if all values are zero
            if all(v == 0 for v in model_tps):
                continue
            
            # Plot the bars
            bars = ax.bar(x_positions + offset_multiplier[j]*width, model_tps, 
                   width, label=strat, color=strategy_colors.get(strat))
            
            # Add value labels on bars
            for k, bar in enumerate(bars):
                if model_tps[k] > 0:  # Only label non-zero bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                            f'{height:.0f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_title(f'Sequence Length = {sl}', fontsize=14)
        ax.set_ylabel('Tokens Per Second (TPS)', fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend(title='Strategy', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.suptitle('Comprehensive TPS Comparison by Model, Strategy, and Sequence Length', 
                 fontsize=18, y=0.98)
    plt.xlabel('Model', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300)
        
    plt.show()

def compare_strategies(df, strategies=None, output_dir=None, figsize=(14, 8)):
    """
    Create a comparison plot focusing on specific parallelism strategies.
    
    Args:
        df (pd.DataFrame): DataFrame with TPS data
        strategies (list): List of strategies to compare (if None, use all strategies)
        output_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Clean strategy names for better display
    df_plot = df.copy()
    df_plot['strategy_display'] = df_plot['strategy'].apply(get_strategy_display_name)
    
    # Filter by strategies if specified
    if strategies:
        filtered_df = df_plot[df_plot['strategy'].isin(strategies)]
    else:
        filtered_df = df_plot
    
    # Group by model and strategy
    grouped_df = filtered_df.groupby(['model', 'strategy_display']).agg({
        'tps': 'mean'
    }).reset_index()
    
    # Plot using seaborn for better styling
    ax = sns.barplot(x='model', y='tps', hue='strategy_display', data=grouped_df)
    
    plt.title('Strategy Comparison Across Models', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Tokens Per Second (TPS)', fontsize=14)
    plt.xticks(rotation=25)
    plt.legend(title='Strategy', fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add TPS values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300)
        
    plt.show()

def plot_speedup(df, baseline_strategy="_1f1b", output_dir=None, figsize=(14, 8)):
    """
    Create a bar plot showing speedup relative to a baseline strategy.
    
    Args:
        df (pd.DataFrame): DataFrame with TPS data
        baseline_strategy (str): The strategy to use as baseline for speedup calculation
        output_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size (width, height)
    """
    # Clean strategy names for better display
    df_plot = df.copy()
    df_plot['strategy_display'] = df_plot['strategy'].apply(get_strategy_display_name)
    
    # Calculate baseline TPS by model
    baseline_df = df_plot[df_plot['strategy'] == baseline_strategy].groupby('model').agg({
        'tps': 'mean'
    }).reset_index()
    
    baseline_df = baseline_df.rename(columns={'tps': 'baseline_tps'})
    
    # Calculate speedup for each model and strategy
    # Group by model and strategy
    grouped_df = df_plot.groupby(['model', 'strategy', 'strategy_display']).agg({
        'tps': 'mean'
    }).reset_index()
    
    # Merge with baseline
    merged_df = pd.merge(grouped_df, baseline_df, on='model')
    
    # Calculate speedup
    merged_df['speedup'] = merged_df['tps'] / merged_df['baseline_tps']
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Remove baseline from plot (speedup = 1)
    plot_df = merged_df[merged_df['strategy'] != baseline_strategy]
    
    ax = sns.barplot(x='model', y='speedup', hue='strategy_display', data=plot_df)
    
    plt.title(f'Speedup Relative to {baseline_strategy.replace("_", "")}', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Speedup (x)', fontsize=14)
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xticks(rotation=25)
    plt.legend(title='Strategy', fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add speedup values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'speedup.png'), dpi=300)
        
    plt.show()

def plot_models_best_config(df, output_dir=None, figsize=(5.5, 4.5)):
    """
    Create a specialized bar plot comparing the models with optimal configurations,
    formatted for a single-column academic paper.
    - For each model, use the configuration that produces the highest TPS
    
    Args:
        df (pd.DataFrame): DataFrame with TPS data
        output_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size (width, height) in inches for paper
    """
    # Create a copy of the dataframe for manipulation
    df_plot = df.copy()
    df_plot['strategy_display'] = df_plot['strategy'].apply(get_strategy_display_name)
    
    # Rename the models to display format
    df_plot['model_display'] = df_plot['model'].replace({
        'qwen_0_5b': 'qwen_0.5b',
        'qwen_1_5b': 'qwen_1.5b',
        'llama3_8b': 'llama3_8b'
    })
    
    # Initialize llama3_df as empty DataFrame
    llama3_df = pd.DataFrame(columns=df_plot.columns)
    
    # Check if llama3_8b data exists
    if 'llama3_8b' in df_plot['model'].unique():
        # 요청에 따라 batch_size를 8로 고정
        target_bs = 8
        
        # 시퀀스 길이는 데이터에서 최적값 찾기
        llama3_data = df_plot[df_plot['model'] == 'llama3_8b']
        llama3_best_sl = llama3_data.loc[llama3_data['tps'].idxmax()]['sequence_length']
        
        # 필터링 (배치 사이즈 8 고정)
        llama3_df = df_plot[(df_plot['model'] == 'llama3_8b') & 
                         (df_plot['sequence_length'] == llama3_best_sl)].copy()
        
        # 원하는 배치 사이즈가 없으면 있는 데이터 사용
        if len(llama3_df[llama3_df['batch_size'] == target_bs]) == 0:
            print(f"Warning: No llama3_8b data with batch_size={target_bs}, using available data")
            llama3_best = llama3_df.loc[llama3_df['tps'].idxmax()]
        else:
            # 배치 사이즈 8로 필터
            llama3_df = llama3_df[llama3_df['batch_size'] == target_bs].copy()
            llama3_best = llama3_df.loc[llama3_df['tps'].idxmax()]
        
        print(f"Using config for llama3_8b: seq_len={llama3_best['sequence_length']}, batch_size={llama3_best['batch_size']}")
        print(f"Llama3 data points with this config: {len(llama3_df)}")
    else:
        print("No llama3_8b data found in the dataset")
    
    # Find best configs for qwen models
    qwen_05b_best = df_plot[df_plot['model'] == 'qwen_0_5b'].loc[df_plot[df_plot['model'] == 'qwen_0_5b']['tps'].idxmax()]
    qwen_15b_best = df_plot[df_plot['model'] == 'qwen_1_5b'].loc[df_plot[df_plot['model'] == 'qwen_1_5b']['tps'].idxmax()]
    
    # Create dataframes with best configs
    qwen_05b_df = df_plot[(df_plot['model'] == 'qwen_0_5b') & 
                         (df_plot['batch_size'] == qwen_05b_best['batch_size']) & 
                         (df_plot['sequence_length'] == qwen_05b_best['sequence_length'])].copy()
    
    qwen_15b_df = df_plot[(df_plot['model'] == 'qwen_1_5b') & 
                         (df_plot['batch_size'] == qwen_15b_best['batch_size']) & 
                         (df_plot['sequence_length'] == qwen_15b_best['sequence_length'])].copy()
    
    # Combine the filtered dataframes
    combined_df = pd.concat([llama3_df, qwen_05b_df, qwen_15b_df])
    print(f"Combined data count: {len(combined_df)}")
    print(f"Model counts: {combined_df['model'].value_counts().to_dict()}")
    
    # Add configuration labels to model names
    combined_df['model_config'] = combined_df.apply(
        lambda row: f"{row['model_display']}\n(seq_len={row['sequence_length']}, bs={8 if row['model'] == 'llama3_8b' else row['batch_size']})", 
        axis=1
    )
    
    # Set paper-friendly style
    plt.style.use('seaborn-v0_8-paper')  # Academic paper style 
    plt.figure(figsize=figsize, dpi=300)
    
    # Use a vibrant, academic-friendly color palette
    palette = sns.color_palette('viridis', n_colors=len(combined_df['strategy_display'].unique()))
    
    # Create the plot with the color palette
    ax = sns.barplot(data=combined_df, x='model_config', y='tps', hue='strategy_display', palette=palette)
    
    # Set paper-friendly labels with smaller fonts
    plt.title('Model Performance Comparison', fontsize=10)
    plt.xlabel('Model Configuration', fontsize=7)
    plt.ylabel('Tokens Per Second (TPS)', fontsize=7)
    
    # 가로(일자)로 레이블 표시 - 범례를 아래에 배치
    plt.legend(title='Strategy', fontsize=6, title_fontsize=7, 
               loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=len(combined_df['strategy_display'].unique()), frameon=True)
    
    # Add grid but make it subtle
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add TPS values with vertical labels for consistency
    for container in ax.containers:
        # Apply vertical bar_label
        ax.bar_label(
            container, 
            fmt='%.0f', 
            fontsize=5, 
            padding=2,
            rotation=90
        )
    
    # Format tick labels for better readability
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the legend
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save in high resolution for publication
        plt.savefig(os.path.join(output_dir, 'models_best_config.png'), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'models_best_config.pdf'), format='pdf', bbox_inches='tight')
    
    plt.show()
    
    # Print the best configurations
    print(f"Best config for qwen_0.5b: seq_len={qwen_05b_best['sequence_length']}, batch_size={qwen_05b_best['batch_size']}")
    print(f"Best config for qwen_1.5b: seq_len={qwen_15b_best['sequence_length']}, batch_size={qwen_15b_best['batch_size']}")
    
    return combined_df


def plot_qwen_models_comparison(df, output_dir=None, figsize=(5.5, 7)):
    """
    Create a specialized plot comparing qwen_0.5b and qwen_1.5b models across different
    sequence lengths, batch sizes, and strategies, formatted for a single-column academic paper.
    
    Args:
        df (pd.DataFrame): DataFrame with TPS data
        output_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size (width, height) in inches for paper
    """
    # Filter only the qwen models
    df_qwen = df[df['model'].isin(['qwen_0_5b', 'qwen_1_5b'])].copy()
    
    # Rename for better display
    df_qwen['model_display'] = df_qwen['model'].replace({
        'qwen_0_5b': 'qwen_0.5b',
        'qwen_1_5b': 'qwen_1.5b'
    })
    
    # Add display strategy names with proper formatting
    df_qwen['strategy_display'] = df_qwen['strategy'].apply(get_strategy_display_name)
    
    # Get unique values for subplot organization
    models = df_qwen['model_display'].unique()
    sequence_lengths = sorted(df_qwen['sequence_length'].unique())
    
    # Set paper-friendly style
    plt.style.use('seaborn-v0_8-paper')  # Academic paper style
    
    # Create a flattened 1x4 grid layout
    n_rows = 1
    # Calculate total number of subplots (models x sequence_lengths)
    n_cols = len(models) * len(sequence_lengths)
    
    # Adjust figure size for the 1x4 layout - make it wider and taller to accommodate spacing
    # Make graphs about 1.3 times wider as requested
    adjusted_figsize = (figsize[0] * 2.0, figsize[1] * 0.7)
    
    # Create the figure and grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=adjusted_figsize, dpi=300, sharey=True)
    
    # Ensure axes is 1D for easier indexing
    if n_cols == 1:
        axes = np.array([axes])
    
    # Define a vibrant color palette for strategies
    n_strategies = len(df_qwen['strategy_display'].unique())
    palette = sns.color_palette('viridis', n_colors=n_strategies)
    
    # Track all unique strategies for the common legend
    all_strategies = None
    all_bars = []
    
    # Iterate through models and sequence lengths to create subplots in a single row
    subplot_idx = 0
    for model in models:
        for sl in enumerate(sequence_lengths):
            sl = sl[1]  # Get the actual sequence length value from the enumerate tuple
            
            # Filter data for this specific model and sequence length
            model_sl_data = df_qwen[(df_qwen['model_display'] == model) & 
                                    (df_qwen['sequence_length'] == sl)]
            
            # Create barplot for this subplot with consistent palette
            sns.barplot(
                data=model_sl_data, 
                x='batch_size', 
                y='tps', 
                hue='strategy_display', 
                ax=axes[subplot_idx],
                palette=palette
            )
            
            # Clean up and format the subplot
            axes[subplot_idx].set_title(f"{model}, seq_len={sl}", fontsize=9)
            axes[subplot_idx].tick_params(labelsize=7)
            axes[subplot_idx].set_xlabel("Batch Size", fontsize=7)
            
            # Only add y-label to the leftmost subplot
            if subplot_idx == 0:
                axes[subplot_idx].set_ylabel("TPS", fontsize=8)
            else:
                axes[subplot_idx].set_ylabel("")
            
            # Remove individual subplot legends
            if axes[subplot_idx].get_legend():
                axes[subplot_idx].get_legend().remove()
            
            # Add value labels with vertical orientation for consistency
            for container in axes[subplot_idx].containers:
                # Apply vertical bar_label
                axes[subplot_idx].bar_label(
                    container, 
                    fmt='%.0f', 
                    fontsize=5, 
                    padding=2,
                    rotation=90
                )
            
            # Keep track of legend objects for later
            if len(axes[subplot_idx].containers) > 0 and (all_strategies is None or len(all_strategies) == 0):  # First non-empty subplot
                all_bars = axes[subplot_idx].containers
                all_strategies = model_sl_data['strategy_display'].unique().tolist()
                
            subplot_idx += 1
    
    # Add common legend at the bottom in a single horizontal row
    if all_bars:
        fig.legend(
            handles=all_bars,
            labels=all_strategies,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),  # Moved closer to the batch size labels
            ncol=len(all_strategies),  # Make legend fully horizontal
            fontsize=7
        )
    
    # Set a common title with slightly larger font
    fig.suptitle("Performance Comparison of qwen Models", fontsize=10, y=0.98)
    
    # Adjust spacing - reduce bottom margin to bring legend closer to batch size labels
    plt.subplots_adjust(top=0.88, bottom=0.15, wspace=0.3, hspace=0.2)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save in high resolution for publication (both PNG and PDF)
        plt.savefig(os.path.join(output_dir, 'qwen_models_comparison.png'), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'qwen_models_comparison.pdf'), format='pdf', bbox_inches='tight')
    
    plt.show()
    
    return df_qwen


def main():
    """Main function to generate all plots"""
    # Set paths - use fixed paths for the existing directory structure
    log_dir = '/home2/yeonjae/tp_partition/torchtitan/logs'
    plot_dir = '/home2/yeonjae/tp_partition/torchtitan/plots'
    
    # Create plots directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Loading TPS data...")
    # Include all strategies including FSDP (_zero) and TP strategies
    df = load_tps_data(log_dir, include_zero=True)
    
    # Make sure we have data for all models and strategies
    print("\nChecking for missing data...")
    if 'llama3_8b' not in df['model'].unique():
        print("WARNING: llama3_8b model data is missing!")
    
    # Check for all parallelism strategies
    expected_strategies = ['_1f1b', '_I1f1b', '_tp_1f1b', '_tp_I1f1b', '_tp', '_zero']
    for strategy in expected_strategies:
        if strategy not in df['strategy'].unique():
            print(f"WARNING: {strategy} strategy data is missing!")
    
    print("\nGenerating specialized plots...")
    
    try:
        # Generate the specialized plots requested by user
        models_df = plot_models_best_config(df, output_dir=plot_dir)
        print("✓ Successfully generated models_best_config plot")
    except Exception as e:
        print(f"ERROR generating models_best_config: {e}")
        
    try:
        # Generate Qwen comparison plot
        qwen_df = plot_qwen_models_comparison(df, output_dir=plot_dir)
        print("✓ Successfully generated qwen_models_comparison plot")
    except Exception as e:
        print(f"ERROR generating qwen_models_comparison: {e}")
    
    print("\nSpecialized plots saved to", plot_dir)
    print("1. models_best_config.png - Comparison of all models with optimal configurations")
    print("2. qwen_models_comparison.png - Detailed comparison of qwen models across all parameters")
    
    return df  # Return dataframe for potential further analysis

if __name__ == "__main__":
    main()
