# COMP6704 Individual Project - Chain-of-Thought (CoT) Model Experiment Framework

This directory contains training and evaluation code for various Chain-of-Thought (CoT) models based on GPT-2, exploring different approaches to reasoning modeling.

## Directory Structure

```
COMP6704_Individual/
├── bash/           # Training and evaluation scripts
├── configs/        # Configuration files
├── scripts/        # Python training and evaluation scripts
├── src/           # Source code
│   ├── data_processing/    # Data processing modules
│   ├── models/            # Model definitions
│   └── utils/             # Utility functions
└── README.md      # This document
```

## Quick Start

### Install Dependencies
```bash
pip install torch transformers pyyaml tqdm tensorboard
```

## Model Types

This project implements the following model architectures:

### 1. Vanilla CoT (Standard Chain-of-Thought)
**Description**: Standard chain-of-thought model with complete reasoning steps
- **Config File**: `configs/gpt2_vanilla.yaml`
- **Training Script**: `bash/train_vanilla.sh`
- **Evaluation Script**: `bash/eval_gpt2_vanilla.sh`
- **Features**: 
  - Input format: Question + Steps + Answer
  - Supervised learning on both reasoning steps and answer
  - Generates complete reasoning process

**Training Command**:
```bash
bash COMP6704_Individual/bash/train_vanilla.sh
```

**Evaluation Command**:
```bash
bash COMP6704_Individual/bash/eval_gpt2_vanilla.sh
```

---

### 2. Baseline (Explicit Start Token)
**Description**: Latent token-based chain-of-thought model with explicit start token
- **Config Files**: 
  - `configs/gpt2_baseline_start_explicit.yaml`
  - `configs/gpt2_baseline_start_explicit_progressive.yaml`
- **Training Scripts**: 
  - `bash/train_gpt2_baseline_startexp.sh`
  - `bash/train_gpt2_baseline_progressive.sh`
- **Evaluation Script**: `bash/eval_latent.sh`
- **Features**:
  - Uses latent tokens to represent reasoning process
  - Loss computed only on answer
  - Progressive version supports gradual training

**Training Commands**:
```bash
# Standard version
bash COMP6704_Individual/bash/train_gpt2_baseline_startexp.sh

# Progressive version
bash COMP6704_Individual/bash/train_gpt2_baseline_progressive.sh
```

**Evaluation Command**:
```bash
bash COMP6704_Individual/bash/eval_latent.sh
```

---

### 3. Reconstruction Distillation
**Description**: Latent reasoning with Reconstruction loss.
- **Config File**: `configs/gpt2_simcot.yaml`
- **Training Script**: `bash/train_gpt2_simcot.sh`
- **Evaluation Script**: `bash/eval_gpt2_simcot.sh`
- **Features**:
  - Uses contrastive learning objective
  - Enhances latent token representation quality
  - Supports continuing from vanilla model

**Training Command**:
```bash
bash COMP6704_Individual/bash/train_gpt2_simcot.sh
```

Optional parameters:
```bash
# Custom configuration
bash COMP6704_Individual/bash/train_gpt2_simcot.sh --config configs/gpt2_simcot.yaml --gpus 2 --gpu-ids 0,2
```

**Evaluation Command**:
```bash
bash COMP6704_Individual/bash/eval_gpt2_simcot.sh
```

---

### 4. Curriculum Learning
**Description**: Progressive increase in number of latent tokens via curriculum learning
- **Config Files**: 
  - `configs/gpt2_curriculum_start_exp_noreset.yaml`
  - `configs/gpt2_curriculum_start_exp.yaml`
- **Training Script**: `bash/train_gpt2_curriculum.sh`
- **Evaluation Script**: `bash/eval_gpt2_curriculum_batch.sh`
- **Features**:
  - Starts with few latent tokens, gradually increases
  - Fixed number of epochs per stage
  - Supports optimizer reset or maintaining state

**Training Command**:
```bash
bash COMP6704_Individual/bash/train_gpt2_curriculum.sh
```

**Evaluation Commands**:
```bash
# Batch evaluate all checkpoints
bash COMP6704_Individual/bash/eval_gpt2_curriculum_batch.sh

# Custom evaluation
bash COMP6704_Individual/bash/eval_gpt2_curriculum_batch.sh \
    --config configs/gpt2_curriculum_start_exp.yaml \
    --checkpoint-dir results/gpt2_curriculum_exp_noreset25_grad \
    --mode training \
    --gpu 5
```

Evaluation modes:
- `training`: Use the number of latent tokens from training
- `max`: Use maximum latent tokens for all checkpoints

---

### 5. Compression Model
**Description**: Compress vanilla CoT to latent tokens using MSE loss
- **Config File**: `configs/gpt2_compression.yaml`
- **Training Script**: `bash/train_gpt2_compression.sh`
- **Evaluation Script**: `bash/eval_gpt2_compression.sh`
- **Features**:
  - Teacher-student architecture
  - Uses pretrained vanilla model as teacher
  - Student model learns compressed representation

**Training Command**:
```bash
bash COMP6704_Individual/bash/train_gpt2_compression.sh
```

**Evaluation Command**:
```bash
bash COMP6704_Individual/bash/eval_gpt2_compression.sh
```

---

### 6. Answer-Only
**Description**: Directly generates answer from question, no reasoning steps
- **Config File**: `configs/gpt2_answer_only.yaml`
- **Training Script**: `bash/train_gpt2_answer_only.sh`
- **Evaluation Script**: `bash/eval_gpt2_answer_only.sh`
- **Features**:
  - Simplest baseline model
  - Input: Question + "### answer"
  - Output: Direct numerical answer

**Training Command**:
```bash
bash COMP6704_Individual/bash/train_gpt2_answer_only.sh
```

**Evaluation Command**:
```bash
bash COMP6704_Individual/bash/eval_gpt2_answer_only.sh
```

---

## Configuration File Explanation

Each configuration file contains the following key parameters:

### General Parameters
- `model_name`: Base model name (e.g., "gpt2")
- `data_path`: Dataset path
- `output_dir`: Output directory
- `batch_size`: Batch size
- `max_epochs`: Maximum training epochs
- `learning_rate`: Learning rate
- `warmup_ratio`: Warmup ratio

### Latent Model Specific Parameters
- `num_latent_tokens`: Number of latent tokens
- `latent_dim`: Latent token dimension
- `loss_on`: Loss computation scope ("answer", "all", etc.)

### Curriculum Specific Parameters
- `c_thought`: Number of tokens to increase per stage
- `epochs_per_stage`: Training epochs per stage
- `max_num_latent`: Maximum number of latent tokens

### Compression Specific Parameters
- `teacher_checkpoint`: Teacher model checkpoint path
- `compression_weight`: Compression loss weight

### Modifying Configuration
1. Copy configuration file
2. Modify relevant parameters
3. Specify new config in training script

Example:
```bash
bash COMP6704_Individual/bash/train_gpt2_simcot.sh --config my_config.yaml
```

---

## Training Details

### Multi-GPU Training
All scripts support multi-GPU training by default using PyTorch DDP:

```bash
# 2 GPU example (modify in script)
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train.py \
    configs/your_config.yaml
```

### Single GPU Training
Modify `NUM_GPUS=1` parameter in script, or use directly:
```bash
python scripts/train.py configs/your_config.yaml
```

### Training Monitoring
Training process automatically logs to TensorBoard:
```bash
tensorboard --logdir results/your_experiment/logs
```

### Checkpoint Saving
- Saves checkpoint at end of each epoch
- Saves best model on validation set
- Format: `checkpoint_epoch{N}.pt` and `best_model.pt`

---

## Evaluation Details

### Evaluation Metrics
All model evaluations include the following metrics:
- **Accuracy**: Proportion of exact answer matches
- **Loss**: Cross-entropy loss
- **Perplexity**: exp(loss)

### Evaluation Output
Evaluation results are saved in JSON format:
```json
{
    "accuracy": 0.75,
    "loss": 1.23,
    "perplexity": 3.42,
    "examples": [...]
}
```

### Batch Evaluation
Curriculum model provides batch evaluation functionality, automatically evaluating all checkpoints and generating summary reports.

---

## Core Script Descriptions

### Training Scripts

| Script File | Corresponding Model | Description |
|---------|---------|------|
| `train_vanilla.sh` | Vanilla CoT | Standard chain-of-thought training |
| `train_gpt2_baseline_startexp.sh` | Baseline | Explicit start token |
| `train_gpt2_baseline_progressive.sh` | Baseline Progressive | Progressive training |
| `train_gpt2_simcot.sh` | Reconstruction | Reconstruction distillation |
| `train_gpt2_curriculum.sh` | Curriculum | Curriculum learning |
| `train_gpt2_compression.sh` | Compression | Compression model |
| `train_gpt2_answer_only.sh` | Answer-Only | Answer-only model |

### Evaluation Scripts

| Script File | Corresponding Model | Description |
|---------|---------|------|
| `eval_gpt2_vanilla.sh` | Vanilla CoT | Standard model evaluation |
| `eval_latent.sh` | Baseline | Latent model evaluation |
| `eval_gpt2_simcot.sh` | Reconstruction | Reconstruction evaluation |
| `eval_gpt2_curriculum_batch.sh` | Curriculum | Batch evaluation |
| `eval_gpt2_compression.sh` | Compression | Compression model evaluation |
| `eval_gpt2_answer_only.sh` | Answer-Only | Answer-only evaluation |

### Python Scripts

| Script File | Function |
|---------|------|
| `scripts/train.py` | Unified training script (supports multiple models) |
| `scripts/train_answer_only.py` | Answer-Only specific training |
| `scripts/evaluate.py` | General evaluation script |
| `scripts/evaluate_answer_only.py` | Answer-Only evaluation |
| `scripts/evaluate_compression.py` | Compression evaluation |
| `scripts/evaluate_curriculum.py` | Curriculum evaluation |
| `scripts/summarize_curriculum_results.py` | Curriculum results summary |

---

## Source Code Structure

### `src/data_processing/`
Data processing module, contains various datasets and data loaders:
- `dataset.py`: Base dataset (GSM8k with latent tokens)
- `vanilla_dataset.py`: Vanilla CoT dataset
- `answer_only_dataset.py`: Answer-Only dataset
- `compression_dataset.py`: Compression dataset
- `curriculum_dataset.py`: Curriculum dataset
- `preprocessor.py`: Data preprocessing utilities

### `src/models/`
Model definition module:
- `gpt2.py`: Base Latent GPT-2 model
- `gpt2_simcot.py`: SIM-CoT model (with similarity constraints)
- `gpt2_compression.py`: Compression model (with reconstruction loss)
- `llama.py`: LLaMA version implementation (experimental)

### `src/utils/`
Utility functions module:
- `utils.py`: General utility functions (setup, logging, saving, etc.)
- `gradient_tracker.py`: Gradient tracking utilities

---

## Experiment Workflow

### Typical Experiment Pipeline

1. **Baseline Experiments** - First train Vanilla and Answer-Only as baselines
```bash
# Vanilla baseline
bash bash/train_vanilla.sh
bash bash/eval_gpt2_vanilla.sh

# Answer-Only baseline
bash bash/train_gpt2_answer_only.sh
bash bash/eval_gpt2_answer_only.sh
```

2. **Latent Model Experiments** - Train basic Latent model
```bash
bash bash/train_gpt2_baseline_startexp.sh
bash bash/eval_latent.sh
```

3. **Improvement Methods** - Try different improvement strategies
```bash
# Curriculum Learning
bash bash/train_gpt2_curriculum.sh
bash bash/eval_gpt2_curriculum_batch.sh

# SIM-CoT
bash bash/train_gpt2_simcot.sh
bash bash/eval_gpt2_simcot.sh

# Compression
bash bash/train_gpt2_compression.sh
bash bash/eval_gpt2_compression.sh
```

4. **Results Analysis** - Compare performance of different methods

---

### Dataset
- **GSM8k**: Grade school math word problems dataset
- Contains approximately 8,000 training problems and 1,000 test problems
