#!/usr/bin/env python3
import os
import re
import glob

def calculate_average_middle_tps(log_file_path):
    """
    Calculate the average TPS from the middle 6 steps out of 10 steps in a log file.
    
    Args:
        log_file_path (str): Path to the log file
    
    Returns:
        float: Average TPS of the middle 6 steps, or None if not enough data
    """
    # Read the log file
    with open(log_file_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # Extract TPS values using regex
    # Pattern looks for step number followed by TPS value
    pattern = r"step:\s+(\d+).*?tps:\s+([\d,]+)"
    matches = re.findall(pattern, content)
    
    # Sort matches by step number and extract TPS values
    step_tps_pairs = [(int(step), float(tps.replace(',', ''))) for step, tps in matches]
    step_tps_pairs.sort()  # Sort by step number
    
    # Group TPS values by step number (in case multiple ranks report the same step)
    step_to_tps = {}
    for step, tps in step_tps_pairs:
        if step not in step_to_tps:
            step_to_tps[step] = []
        step_to_tps[step].append(tps)
    
    # Get average TPS for each step (across all ranks)
    steps = []
    for step in sorted(step_to_tps.keys()):
        avg_tps = sum(step_to_tps[step]) / len(step_to_tps[step])
        steps.append((step, avg_tps))
    
    # If we have at least 10 steps, calculate average of middle 6
    if len(steps) >= 10:
        # Take steps 3, 4, 5, 6, 7, 8 (middle 6 out of 10)
        middle_six_steps = steps[2:8]
        middle_six_tps = [tps for _, tps in middle_six_steps]
        return sum(middle_six_tps) / len(middle_six_tps)
    else:
        print(f"Warning: Not enough steps found in {log_file_path}. Found {len(steps)} steps.")
        return None

def process_model_logs():
    """
    Process logs for each model and save individual TPS values for each subfolder to corresponding txt files.
    """
    model_folders = ["llama3_8b", "qwen_0_5b", "qwen_1_5b"]
    base_path = "/home2/yeonjae/tp_partition/torchtitan/logs"
    
    for model in model_folders:
        model_path = os.path.join(base_path, model)
        
        if not os.path.exists(model_path):
            print(f"Warning: Model folder {model_path} does not exist. Skipping.")
            continue
        
        # Find all log files in the model directory and its subdirectories
        folder_tps_map = {}
        
        # Walk through all subdirectories
        for root, _, files in os.walk(model_path):
            for file in files:
                if file == "log.txt":
                    log_file = os.path.join(root, file)
                    # Get the folder name (the last component of the path containing log.txt)
                    folder_name = os.path.basename(os.path.dirname(log_file))
                    
                    # Calculate average TPS for this log file
                    avg_tps = calculate_average_middle_tps(log_file)
                    
                    if avg_tps is not None:
                        folder_tps_map[folder_name] = avg_tps
        
        if not folder_tps_map:
            print(f"Warning: No valid TPS data found for {model}")
            continue
            
        print(f"Processed {len(folder_tps_map)} folders for model {model}...")
        
        # Write result to a txt file named after the model
        output_file = os.path.join(model_path, f"{model}.txt")
        with open(output_file, 'w') as f:
            for folder, tps in sorted(folder_tps_map.items()):
                f.write(f"{folder}: {tps:.2f}\n")
        
        print(f"Saved results to {output_file}")

if __name__ == "__main__":
    process_model_logs()
