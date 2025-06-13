## 🔧 Setup Instructions

Follow these steps to set up the environment for training:

### 1. Clone and Install PyTorch

```bash
cd pytorch
conda install -y cmake ninja
conda install -y rust
pip install -r requirements.txt
conda install -y -c pytorch magma-cuda124
make triton
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-"$(dirname $(which conda))/../"}:${CMAKE_PREFIX_PATH}"
python setup.py develop
```

### 2. Clone and Install TorchTitan

```bash
cd ../torchtitan
pip install -r requirements.txt
```

### 3. Download Tokenizer from Hugging Face

```bash
huggingface-cli login
python scripts/download_tokenizer.py \
  --repo_id meta-llama/Meta-Llama-3.1-8B \
  --tokenizer_path "original"
```