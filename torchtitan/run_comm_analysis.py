import subprocess
import os

def run_analysis(configs, output_file):
    with open(output_file, 'w') as f:
        f.write("# Communication Volume Analysis for Different Model Configurations\n\n")
        
        for config in configs:
            model_name = config['name']
            hidden_size = config['hidden_size']
            intermediate_size = config['intermediate_size']
            tp_size = config['tp_size']
            pp_size = config['pp_size']
            if config["name"]=="llama3_8b":
                seq_length = 1024
                batch_size = 8
            elif config["name"]=="qwen_1_5b":
                seq_length = 2048
                batch_size = 16
            elif config["name"]=="qwen_0_5b":
                seq_length = 2048
                batch_size = 32
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Model: {model_name}, TP Size: {tp_size}, PP Size: {pp_size}\n")
            f.write(f"Hidden Size: {hidden_size}, Intermediate Size: {intermediate_size}\n")
            f.write(f"Sequence Length: {seq_length}, Batch Size: {batch_size}\n")
            f.write(f"{'-'*80}\n\n")
            
            cmd = [
                "python", "simplified_comm_calculator.py",
                "--hidden_size", str(hidden_size),
                "--intermediate_size", str(intermediate_size),
                "--seq_length", str(seq_length),
                "--batch_size", str(batch_size),
                "--tp_size", str(tp_size)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            f.write(result.stdout)
            f.write("\n")

# Define model configurations
configs = [
    # llama3_8b configurations
    {
        'name': 'llama3_8b',
        'hidden_size': 4096,
        'intermediate_size': 14336,  # approximation based on ffn_dim_multiplier
        'tp_size': 1,
        'pp_size': 4,
    },
    {
        'name': 'llama3_8b',
        'hidden_size': 4096,
        'intermediate_size': 14336,
        'tp_size': 2,
        'pp_size': 2,
    },
    {
        'name': 'llama3_8b',
        'hidden_size': 4096,
        'intermediate_size': 14336,
        'tp_size': 4,
        'pp_size': 1,
    },
    
    # qwen_1_5b configurations
    {
        'name': 'qwen_1_5b',
        'hidden_size': 1536,
        'intermediate_size': 8960,
        'tp_size': 1,
        'pp_size': 4,
    },
    {
        'name': 'qwen_1_5b',
        'hidden_size': 1536,
        'intermediate_size': 8960,
        'tp_size': 2,
        'pp_size': 2,
    },
    
    # qwen_0_5b configurations
    {
        'name': 'qwen_0_5b',
        'hidden_size': 896,
        'intermediate_size': 4864,
        'tp_size': 1,
        'pp_size': 4,
    },
    {
        'name': 'qwen_0_5b',
        'hidden_size': 896,
        'intermediate_size': 4864,
        'tp_size': 2,
        'pp_size': 2,
    },
]

# Run analysis and save results
output_file = "communication_analysis_results.txt"
run_analysis(configs, output_file)
print(f"Analysis complete. Results saved to {output_file}")

# Print a summary table at the end
print("\nGenerating summary table...")
with open(output_file, 'a') as f:
    f.write("\n\n" + "="*80 + "\n")
    f.write("SUMMARY TABLE\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Model':<12} {'Strategy':<10} {'TP Comm MB/block':<20} {'Activation MB':<15} {'Gradient MB':<15}\n")
    f.write("-"*75 + "\n")

# Open the file again to extract relevant data for the summary table
with open(output_file, 'r') as f:
    lines = f.readlines()

# Process the data for the summary table
current_model = ""
current_tp = 0
current_pp = 0
summary_data = []

for i, line in enumerate(lines):
    if "Model:" in line and ", TP Size:" in line:
        parts = line.split(", ")
        model = parts[0].replace("Model: ", "")
        tp_size = int(parts[1].replace("TP Size: ", ""))
        pp_size = int(parts[2].replace("PP Size: ", ""))
        current_model = model
        current_tp = tp_size
        current_pp = pp_size
        
    elif "Total TP Communication:" in line:
        total_tp_comm = float(line.split("MB")[0].replace("Total TP Communication: ", "").strip())
        
    elif "Forward Activation Size:" in line:
        activation_size = float(line.split("MB")[0].replace("Forward Activation Size: ", "").strip())
        
    elif "Backward Gradient Size:" in line:
        gradient_size = float(line.split("MB")[0].replace("Backward Gradient Size: ", "").strip())
        
        # After we've found all the data for this configuration, add it to our summary
        if current_model and current_tp >= 0 and current_pp >= 0:
            strategy = f"TP{current_tp}+PP{current_pp}" if current_pp > 0 else f"TP{current_tp}"
            summary_data.append({
                'model': current_model,
                'strategy': strategy,
                'tp_comm': total_tp_comm if current_tp > 1 else 0,
                'activation': activation_size,
                'gradient': gradient_size
            })

# Append the summary table to the output file
with open(output_file, 'a') as f:
    for data in summary_data:
        f.write(f"{data['model']:<12} {data['strategy']:<10} {data['tp_comm']:<20.2f} {data['activation']:<15.2f} {data['gradient']:<15.2f}\n")

print(f"Summary table added to {output_file}")
