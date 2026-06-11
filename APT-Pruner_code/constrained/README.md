# Global-Interaction-Aware Token Pruning for Vision Transformers

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An official implementation of the **Methodology-Aligned ViT Pruning Framework**. This project implements a **Two-Stage** pruning pipeline that combines rapid static priors with Asynchronous Reinforcement Learning (A3C) to achieve efficient and accurate Vision Transformer acceleration under strict budget constraints.

## 📖 Methodology Overview

This framework strictly follows the methodology:

1.  **Stage 1: Rapid Static Prior Extraction**
    * Computes **Effective Fisher Information (EFI)** via virtual gate gradients.
    * Generates a global importance matrix to solve the RL cold-start problem.
    * Normalizes priors using Z-score statistics.

2.  **Stage 2: RL-Driven Dynamic Mask Generation**
    * **Asynchronous A3C**: Uses parallel workers for stable gradient estimation and faster convergence.
    * **Factorized Policy**: A shared parameter network handling variable sequence lengths, taking `[Prior, Global, Local, Pos]` as input state.
    * **Hybrid Reward**: Combines Feature Reconstruction (MSE) and Budget Alignment penalties.
    * **Asymmetric Execution**: Bernoulli sampling during training (exploration) vs. Deterministic **Top-k** during inference (strict constraint).



## 📂 🚀 Quick Start

1. Basic Training (Stage 1 + Stage 2)
Run the A3C training with 4 parallel workers on GPU 0. This will prune vit-base to keep 70% of tokens.

python main_distributed.py \
    --model_name "google/vit-base-patch16-224" \
    --ckpt_dir "./vit-weights/vit-base-patch16-224" \
    --data_dir "/path/to/imagenet" \
    --constraint 0.7 \
    --num_workers 4 \
    --max_steps 500 \
    --gpu 0

2. Ablation Studies
Switch between different reward configurations using the --experiment flag (defined in main_distributed.py):

original_norm (Default): Full hybrid reward with normalization.

v3_norm: Accuracy-only reward (for comparison).

python main_distributed.py \
    --experiment v3_norm \
    --constraint 0.5 \
    --num_workers 8




⚙️ Key Arguments

Argument,Default,Description
--model_name,vit-base...,HuggingFace model ID or path.
--data_dir,Required,Path to ImageNet/ImageNet-100 dataset.
--constraint,0.7,Target token retention ratio (0.0-1.0).
--num_workers,4,Number of parallel A3C workers.
--max_steps,100,Training steps per worker.
--experiment,original_norm,Reward configuration strategy.


🧠 Technical Highlights
Physical Pruning Simulation: The A3CPruningWorker maintains an Origin Index Map to simulate the physical removal of tokens (Gather operation) while keeping track of their original positions for positional encoding alignment.

Shared Optimization: Uses a custom SharedAdam optimizer to synchronize gradients across multiple processes via shared memory.

Strict Budget: Inference uses Top-k selection based on policy logits, ensuring the token count exactly matches the budget constraint.

📊 Outputs
Results are saved in outputs/:

final_mask.pt: The learned binary mask/importance scores.

policy_net.pth: Checkpoint of the trained RL agent.

Training logs with reward statistics.

