import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class ConstrainedPolicyNet(nn.Module):
    """
    对应 Methodology Sec 3.3.1: Factorized Policy (因式分解策略网络)

    核心特性:
    1. 参数共享 (Parameter Shared): 对任意长度序列的每个Token独立处理。
    2. 状态输入: 接收 [B, N, D] 的组合状态 (Prior + Global + Local + Pos)。
    3. Actor-Critic: 同时输出 Actor (Logits) 和 Critic (Value)。
    4. 非对称执行 (Asymmetric Execution): 训练时Bernoulli采样，推理时Top-k约束。
    """

    def __init__(self, d_glob, d_loc, d_pos, hidden_dim=128):
        super().__init__()

        # 状态维度: 1 (Normalized Prior) + d_glob + d_loc + d_pos
        self.input_dim = 1 + d_glob + d_loc + d_pos

        # 共享特征提取器 f_theta
        # 这里的 Linear 是作用在最后一维上的，相当于对每个 Token 独立应用 MLP
        self.shared_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 增加 LayerNorm 提高训练稳定性
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor Head: 输出每个 token 的保留概率 logit l_{t,i}
        self.actor_head = nn.Linear(hidden_dim, 1)

        # Critic Head: 输出全局状态价值 V(s)
        # Critic 需要感知全局信息，因此在 Head 内部会进行聚合
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        前向传播

        Args:
            state: [Batch, N_tokens, Input_Dim] 组合状态向量

        Returns:
            logits: [Batch, N_tokens, 1] 用于构建 Bernoulli 分布
            value: [Batch, 1] 全局状态价值估计
        """
        # 1. 提取特征 [B, N, H]
        features = self.shared_net(state)

        # 2. Actor 输出 Logits [B, N, 1]
        logits = self.actor_head(features)

        # 3. Critic 输出 Value [B, 1]
        # Methodology 中提到 Critic 评估整个状态，这里使用 Mean Pooling 聚合所有 Token 的特征
        global_feat = features.mean(dim=1)
        value = self.critic_head(global_feat)

        return logits, value

    def sample_action(self, logits, target_count=None, training=True):
        """
        非对称动作生成 (Sec 3.3.2)

        Args:
            logits: [Batch, N, 1]
            target_count: int, 目标保留数量 (Top-k 推理时需要)
            training: bool, 训练模式还是推理模式

        Returns:
            mask: [Batch, N, 1] 0/1 Mask
            log_prob: [Batch] 动作的对数概率 (仅训练时有效)
            entropy: [Batch] 熵 (仅训练时有效)
        """
        B, N, _ = logits.shape

        # 确保 CLS Token (Index 0) 的 Logit 非常大，保证始终被选中/保留概率高
        # 注意：这修改了原 logits，不影响梯度回传，但影响采样分布
        # clone防止原地修改报错
        logits = logits.clone()
        logits[:, 0, :] += 100.0

        probs = torch.sigmoid(logits)

        if training:
            # === Training: Bernoulli Sampling ===
            dist = Bernoulli(probs=probs)
            mask = dist.sample()  # [B, N, 1]

            # 计算 Log Prob (Sum over tokens, independent assumption)
            # log P(a|s) = sum log P(a_i|s_i)
            log_prob = dist.log_prob(mask).sum(dim=1).squeeze(-1)  # [B]
            entropy = dist.entropy().sum(dim=1).squeeze(-1)  # [B]

        else:
            # === Inference: Deterministic Top-k ===
            # 严格满足 Budget 约束
            assert target_count is not None, "Inference mode requires target_token_count"

            # 获取 Top-k indices
            # logits.squeeze(-1): [B, N]
            topk_vals, topk_indices = torch.topk(logits.squeeze(-1), k=target_count, dim=1)

            mask = torch.zeros_like(logits).squeeze(-1)  # [B, N]
            mask.scatter_(1, topk_indices, 1.0)
            mask = mask.unsqueeze(-1)  # [B, N, 1]

            log_prob = None
            entropy = None

        return mask, log_prob, entropy


# 这是一个兼容层，如果你的代码其它部分需要 ConstrainedCriticNet 这个名字
class ConstrainedCriticNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # 实际逻辑已集成在 ConstrainedPolicyNet 中
        # 如果需要独立 Critic，可以单独实例化一个网络
        pass


def test_constrained_policy():
    print("🧪 测试符合 Methodology 的 Factorized Policy 网络...")

    # 模拟参数
    B, N = 2, 197
    d_glob, d_loc, d_pos = 64, 64, 32
    target_k = 139

    policy = ConstrainedPolicyNet(d_glob, d_loc, d_pos)

    # 模拟输入状态 [B, N, 1 + dg + dl + dp]
    input_dim = 1 + d_glob + d_loc + d_pos
    state = torch.randn(B, N, input_dim)

    # 前向传播
    logits, value = policy(state)
    print(f"Logits Shape: {logits.shape} (Expect [{B}, {N}, 1])")
    print(f"Value Shape: {value.shape} (Expect [{B}, 1])")

    # 1. 测试训练模式 (Bernoulli)
    print("\n--- Training Mode (Bernoulli) ---")
    mask_train, log_prob, entropy = policy.sample_action(logits, training=True)
    print(f"Mask Shape: {mask_train.shape}")
    print(f"Sample Count (Random): {mask_train[0].sum().item()}")
    print(f"CLS Token Selected: {mask_train[0, 0].item() == 1.0}")

    # 2. 测试推理模式 (Top-k)
    print(f"\n--- Inference Mode (Top-{target_k}) ---")
    mask_eval, _, _ = policy.sample_action(logits, target_count=target_k, training=False)
    actual_k = int(mask_eval[0].sum().item())
    print(f"Target k: {target_k}, Actual k: {actual_k}")
    print(f"CLS Token