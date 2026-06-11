import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedPolicyNet(nn.Module):
    """
    对应 Sec 3.3.1: Factorized Policy & State Formulation
    State components per token:
    1. Normalized Static Prior (1 dim)
    2. Global Context (d_g dim) - Mean pooled from current layer
    3. Local Dynamic Feature (d_l dim) - Specific token feature
    4. Positional Encoding (d_p dim) - Via Origin Index Map
    """

    def __init__(self, d_glob, d_loc, d_pos, hidden_dim=128):
        super(FactorizedPolicyNet, self).__init__()

        # State dimension: 1 (prior) + d_g + d_l + d_p
        self.input_dim = 1 + d_glob + d_loc + d_pos

        # Shared feature extractor f_theta
        self.shared_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor Head: Outputs logit l_{t,i} for Bernoulli probability
        self.actor_head = nn.Linear(hidden_dim, 1)

        # Critic Head: Outputs Value V(s)
        # Note: Critic estimates value based on the aggregated state or mean state
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Args:
            state: [Batch, N_tokens, input_dim]
        Returns:
            logits: [Batch, N_tokens, 1] (For Actor)
            value: [Batch, 1] (For Critic, aggregating token info)
        """
        features = self.shared_net(state)  # [B, N, H]

        # Actor: Independent decision per token
        logits = self.actor_head(features)  # [B, N, 1]

        # Critic: Global value estimate (using mean pooling over tokens)
        # 也可以使用 Attention pooling，这里简化为 Mean
        features_pooled = features.mean(dim=1)  # [B, H]
        value = self.critic_head(features_pooled)  # [B, 1]

        return logits, value