#!/usr/bin/env python3
"""
ViT Hybrid Pruning Framework - A3C Implementation
1. Stage 1: 基于EFI (Effective Fisher Information) 的静态先验提取
2. Stage 2: 基于A3C的动态掩码生成 (RL-Driven Dynamic Mask Generation)

"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Bernoulli
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# 辅助类: 共享优化器 (用于A3C)
# ==========================================
class SharedAdam(torch.optim.Adam):
    """支持多进程共享内存的Adam优化器"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # 初始化状态
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # 共享内存
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


# ==========================================
# Sec 3.3.1: 策略网络 (Factorized Policy)
# ==========================================
class ActorCriticNetwork(nn.Module):
    def __init__(self, d_glob, d_loc, d_pos, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        # State components: Normalized Prior(1) + Global(dg) + Local(dl) + Pos(dp)
        self.input_dim = 1 + d_glob + d_loc + d_pos

        # Shared Encoder f_theta
        self.shared_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor: 输出保留概率 logits (Independent per token)
        self.actor_head = nn.Linear(hidden_dim, 1)

        # Critic: 输出状态价值 Value V(s)
        # Critic 需要评估整体状态，这里简单使用特征均值聚合
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # state shape: [Batch, N_current, Input_Dim]
        features = self.shared_layer(state)  # [B, N, Hidden]

        # Actor Logits
        logits = self.actor_head(features)  # [B, N, 1]

        # Critic Value (Aggregate over tokens to get global state value)
        # Sec 3.3.1 中提到利用 Global Context，这里简化为对特征做 Mean Pooling
        global_feat = features.mean(dim=1)  # [B, Hidden]
        value = self.critic_head(global_feat)  # [B, 1]

        return logits, value


# ==========================================
# Stage 1: 静态先验提取 (Sec 3.1)
# ==========================================
def compute_efi_score(model, dataloader, device, num_batches=20):
    """
    计算 Effective Fisher Information (EFI)
    对应公式 (5): E[(dL/dg)^2]
    """
    logger.info(f"正在计算静态先验 (EFI), 使用 {num_batches} 个Batch...")
    model.eval()

    # 模拟获取 Embedding 层的输出维度
    # 实际应 hook model.vit.embeddings.dropout 或类似层
    # 假设 ViT-Base: 197 tokens (14x14 + 1 cls)
    num_tokens = 197
    importances = []

    # 模拟 Virtual Gate 梯度计算
    # 在真实代码中，你需要 register_hook 到 embedding 层
    # 这里使用随机数模拟计算过程
    for i in range(num_batches):
        # 模拟梯度: grad ~ Normal(0, 1)
        # EFI = grad^2
        mock_grad = torch.randn(num_tokens, device=device)
        efi = mock_grad.pow(2)
        importances.append(efi)

    # 全局平均
    global_importance_I = torch.stack(importances).mean(dim=0)  # [N]

    # 预计算归一化统计量 (Sec 3.3.1 Eq.7)
    prior_stats = {
        'mu': global_importance_I.mean(),
        'sigma': global_importance_I.std()
    }
    logger.info(f"EFI计算完成. Mu={prior_stats['mu']:.4f}, Sigma={prior_stats['sigma']:.4f}")
    return global_importance_I.cpu(), prior_stats


# ==========================================
# Stage 2: A3C Worker (Sec 3.3)
# ==========================================
class PruningWorker(mp.Process):
    def __init__(self, worker_id, args, global_net, optimizer, static_prior, prior_stats, save_lock):
        super(PruningWorker, self).__init__()