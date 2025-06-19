## 🔧 Setup Instructions

Follow these steps to set up the environment for training:

### 1. Clone and Install PyTorch

```bash
cd pytorch              # Enter PyTorch directory
conda install -y cmake ninja    # Install build tools for PyTorch
conda install -y rust    # Install Rust for PyTorch components
pip install -r requirements.txt    # Install PyTorch dependencies
conda install -y -c pytorch magma-cuda124    # Install MAGMA for GPU acceleration
make triton            # Build Triton tensor compiler
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-"$(dirname $(which conda))/../"}:${CMAKE_PREFIX_PATH}"    # Set build environment variable
python setup.py develop    # Build and install PyTorch in development mode
```

### 2. Clone and Install TorchTitan

```bash
cd ../torchtitan        # Enter TorchTitan directory
pip install -r requirements.txt    # Install TorchTitan dependencies
python setup.py develop    # Install TorchTitan in development mode
```

### 3. Download Tokenizer from Hugging Face

```bash
huggingface-cli login    # Log in to Hugging Face account
python scripts/download_tokenizer.py \    # Download Meta-Llama-3.1-8B tokenizer
  --repo_id meta-llama/Meta-Llama-3.1-8B \
  --tokenizer_path "original"
```

# run_utils.py

A lightweight launcher that **applies on-the-fly overrides to any TorchTitan
`*.toml` config, executes `run_train.sh`, and writes a single timestamped log
file per run**.  Ideal for sweeping parallelism settings (TP, PP, FSDP) or
batch/sequence sizes without hand-editing configs.

---

## ✨ Key Features
| Feature | Description |
|---------|-------------|
| **Config patching** | Reads a base `.toml`, maps `section.key`, and injects overrides from a Python dict. |
| **Self-naming folders** | Generates human-readable run folders, e.g. `_bs8_sl1_tp`, so results stay organised. |
| **One-click launch** | Builds CLI flags (`--section.key=value`), sets `CONFIG_FILE`, then calls `run_train.sh`. |
| **Unified log** | Captures command header + `stdout` + `stderr` into `logs/<model>/<run>/log.txt`. |

---

## 🚀 Quick Start

```bash
python run_utils.py
```

### `simplified_comm_calculator.py`

A tiny CLI / helper module that lets you **estimate communication volume** for a single Transformer block under tensor-parallel (TP) and pipeline-parallel (PP) settings.

| Function | What it returns |
|----------|-----------------|
| `calculate_tp_communication_single_block(...)` | MB moved by each TP collective (reduce-scatter / all-gather) in forward and backward. |
| `get_pipeline_block_sizes(...)` | MB of activations exchanged at a PP boundary |

#### Example

```bash
python simplified_comm_calculator.py \
  --hidden_size 4096 \
  --seq_length 2048 \
  --batch_size 32 \
  --tp_size 4
```

### `run_comm_analysis.py`

A tiny driver that calls **`simplified_comm_calculator.py`** for a list of
(model-, TP-, PP-) configurations, then writes:

1. **Raw run logs** for each setup (`communication_analysis_results.txt`)
2. **A one-line summary table** at the end of that file

#### Example

```bash
python run_comm_analysis.py
```

This script was used to generate all communication-volume numbers reported in
the paper.
