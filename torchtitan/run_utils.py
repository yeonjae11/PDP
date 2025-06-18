import subprocess
import toml
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

def overrides_to_filename(overrides: Dict[str, Any]) -> str:
    name = ""
    if overrides.get("pipeline_parallel_degree",0) == 1 and overrides.get("tensor_parallel_degree",0) == 1:
        name += "_zero"
    for key, value in overrides.items():
        if key == "local_batch_size":
            name += f"_bs{value}"
        elif key == "seq_len":
            v = value//1024
            name += f"_sl{v}"
        elif overrides["pipeline_parallel_degree"] != 1 and key == "pipeline_parallel_schedule":
            if value == "1F1B":
                name += "_1f1b"
            elif value == "Interleaved1F1B":
                name += "_I1f1b"
            elif value == "ZBVZeroBubble":
                name += "_zvzb"
            else:
                name += f"_{value}"
        elif key == "tensor_parallel_degree" and overrides["tensor_parallel_degree"] != 1:
            name += f"_tp"
    return name

def run_training_with_overrides(
    base_config_path: str,
    overrides: Dict[str, Any],
    run_script_path: str = "./run_train.sh",
    log_dir: str = "run_logs"
):
    folder_name = overrides_to_filename(overrides)
    print("folder_name:", folder_name)
    log_dir = os.path.join(log_dir, folder_name)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file_path = os.path.join(log_dir, "log.txt")

    try:
        config_data = toml.load(base_config_path)
        key_to_section = {
            key: section
            for section, settings in config_data.items()
            for key in settings
        }
    except FileNotFoundError:
        error_msg = f"Error: Config file '{base_config_path}' not found."
        print(error_msg)
        with open(log_file_path, "w") as f:
            f.write(error_msg)
        return
    except Exception as e:
        error_msg = f"Error: toml file read error: {e}"
        print(error_msg)
        with open(log_file_path, "w") as f:
            f.write(error_msg)
        return
    overrides_copy = overrides.copy()
    overrides_copy["dump_folder"] = log_dir
    overrides_copy["save_traces_folder"] = ""
    overrides_copy["save_tb_folder"] = ""
    cli_overrides = []
    for key, value in overrides_copy.items():
        if key not in key_to_section:
            print(f"Warning: Key '{key}' not found in toml file. Ignored.")
            continue
        
        section = key_to_section[key]
        cli_overrides.append(f"--{section}.{key}={value}")

    command = [run_script_path] + cli_overrides
    env = os.environ.copy()
    env["CONFIG_FILE"] = str(Path(base_config_path).resolve())

    header = f"""
============================================================
- exec time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- command: {' '.join(command)}
- config file: {env['CONFIG_FILE']}
- overrides: {overrides}
============================================================
"""
    print(header)

    try:
        process = subprocess.run(
            command,
            check=True,
            text=True,
            env=env,
            capture_output=True
        )
        print("\n✅ done!")

    except FileNotFoundError:
        print(f"Error: run script '{run_script_path}' not found.")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"\nError: run script '{run_script_path}' not found.")
        return
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: script execution failed (Exit Code: {e.returncode})")
        process = e
    
    log_content = f"{header}\n--- STDOUT ---\n{process.stdout}\n--- STDERR ---\n{process.stderr}"
    
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(log_content)
        
    print(f"✅ done! All logs saved to: {log_file_path}")

if __name__ == '__main__':
    toml_path = "./torchtitan/models/llama3/train_configs/llama3_8b.toml"
    overrides_list = []
    tp_overrides = {
        "pipeline_parallel_degree": 1,
        "tensor_parallel_degree": 4,
        "seq_len": 1024,
    }
    overrides_list.append(tp_overrides)
    
    pp_1f1b_overrides = {
        "pipeline_parallel_degree": 4,
        "tensor_parallel_degree": 1,
        "pipeline_parallel_schedule": "1F1B",
        "pipeline_parallel_num_stages_per_rank": 1,
        "seq_len": 1024,
    }
    overrides_list.append(pp_1f1b_overrides)

    pp_interleaved_1f1b_overrides = {
        "pipeline_parallel_degree": 4,
        "tensor_parallel_degree": 1,
        "pipeline_parallel_schedule": "Interleaved1F1B",
        "pipeline_parallel_num_stages_per_rank": 2,
        "seq_len": 1024,
    }
    overrides_list.append(pp_interleaved_1f1b_overrides)

    tp2_pp2_1f1b_overrides = {
        "tensor_parallel_degree": 2,
        "pipeline_parallel_degree": 2,
        "pipeline_parallel_schedule": "1F1B",
        "pipeline_parallel_num_stages_per_rank": 1,
        "seq_len": 1024,
    }
    overrides_list.append(tp2_pp2_1f1b_overrides)

    tp2_pp2_interleaved_1f1b_overrides = {
        "tensor_parallel_degree": 2,
        "pipeline_parallel_degree": 2,
        "pipeline_parallel_schedule": "Interleaved1F1B",
        "pipeline_parallel_num_stages_per_rank": 2,
        "seq_len": 1024,
    }
    overrides_list.append(tp2_pp2_interleaved_1f1b_overrides)

    zero_overrides = {
        "tensor_parallel_degree": 1,
        "pipeline_parallel_degree": 1,
        "seq_len": 1024,
    }
    overrides_list.append(zero_overrides)

    for overrides in overrides_list:
        if overrides != zero_overrides:
            continue
        run_training_with_overrides(
            base_config_path=toml_path,
            overrides=overrides,
            run_script_path="./run_train.sh",
            log_dir="logs/llama3_8b"
        )
    
    batch_sizes = [8, 16, 32]
    toml_path_05b = "./torchtitan/models/llama3/train_configs/qwen_0_5b.toml"
    for override in overrides_list:
        if override == tp_overrides:
            overrides_list.remove(override)
    
    for batch_size in batch_sizes:
        for seq_len in [2048, 4096]:
            if seq_len == 4096 and batch_size == 32:
                continue
            for overrides in overrides_list:
                if overrides != zero_overrides:
                    continue
                tmp_overrides = overrides.copy()
                tmp_overrides["local_batch_size"] = batch_size
                tmp_overrides["seq_len"] = seq_len
                run_training_with_overrides(
                    base_config_path=toml_path_05b,
                    overrides=tmp_overrides,
                    run_script_path="./run_train.sh",
                    log_dir="logs/qwen_0_5b"
                )
    toml_path_1_5b = "./torchtitan/models/llama3/train_configs/qwen_1_5b.toml"
    for batch_size in [8,16]:
        for seq_len in [2048, 4096]:
            if seq_len == 4096 and batch_size == 16:
                continue
            for overrides in overrides_list:
                if overrides != zero_overrides:
                    continue
                tmp_overrides = overrides.copy()
                tmp_overrides["local_batch_size"] = batch_size
                tmp_overrides["seq_len"] = seq_len
                run_training_with_overrides(
                    base_config_path=toml_path_1_5b,
                    overrides=tmp_overrides,
                    run_script_path="./run_train.sh",
                    log_dir="logs/qwen_1_5b"
                )

        