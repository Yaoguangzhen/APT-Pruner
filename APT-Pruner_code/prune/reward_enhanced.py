import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class HybridRewardCalculator:
    """
    对应 Sec 3.3.4: Hybrid Reward Mechanism
    R_t = w_r * R_rec + R_bud
    """

    def __init__(self, w_rec=1.0, w_bud=10.0, budget_ratio=0.7):
        self.w_rec = w_rec
        self.w_bud = w_bud
        self.budget_ratio = budget_ratio

    def compute_reward(self, pruned_features, base_features, origin_index_map, current_N, full_N):
        """
        计算单步奖励

        Args:
            pruned_features: [B, N_t, D] 剪枝后的特征 (E_t)
            base_features: [B, N, D] 原始Base模型特征 (E_base)
            origin_index_map: [B, N_t] 映射 pruned_indices -> original_indices
            current_N: int, 当前保留的token数量 N_t
            full_N: int, 原始token数量 N

        Returns:
            reward: [B]
        """
        device = pruned_features.device
        batch_size = pruned_features.shape[0]

        # 1. Feature Reconstruction Reward (R_rec) [Eq. 15]
        # Align features using origin_index_map
        # Gather base features corresponding to the kept tokens
        # base_features: [B, N, D]
        # map expanded: [B, N_t, D]
        idx_expanded = origin_index_map.unsqueeze(-1).expand(-1, -1, pruned_features.shape[-1])
        base_features_aligned = torch.gather(base_features, 1, idx_expanded)

        # MSE Calculation
        # Normalized by N_t * D
        diff = pruned_features - base_features_aligned
        mse = diff.pow(2).sum(dim=(1, 2)) / (current_N * pruned_features.shape[-1])
        r_rec = -mse  # Negative MSE

        # 2. Budget Alignment Reward (R_bud) [Eq. 16]
        # Target budget k_t
        k_t = int(self.budget_ratio * full_N)
        # Penalty
        budget_penalty = -((current_N - k_t) / full_N) ** 2

        # Total Reward [Eq. 17]
        total_reward = self.w_rec * r_rec + self.w_bud * budget_penalty

        return total_reward

    def compute_terminal_reward(self, loss_pruned, loss_base):
        """
        对应 Eq. 18: Global Terminal Reward
        """
        # R_term = -w_task * (L_pruned - L_base)
        # Using a default w_task=1.0 for simplicity, adjustable
        return -(loss_pruned - loss_base)