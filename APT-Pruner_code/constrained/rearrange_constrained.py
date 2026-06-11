import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


def constrained_a3c_step(
        model,
        dataloader,
        fisher_info,
        token_mask,
        policy_net,
        optimizer,
        config,
        seq_len,
        target_token_count=139,
        gamma=0.99,
        entropy_coef=0.01,
        num_eval_samples=1000
):
    """
    执行一步 A3C 更新 (Constraint-Aware)

    Args:
        policy_net: FactorizedPolicyNet (Actor-Critic)
        optimizer: SharedAdam
    """
    device = next(policy_net.parameters()).device
    policy_net.train()
    model.eval()

    # 1. 状态构建 (State Construction)
    # 状态包含 [Prior, Global, Local, Pos]
    # 这里简化为直接使用 Fisher 和 Mask (需与你的 PolicyNet 输入对齐)
    prior = fisher_info.to(device).view(1, -1, 1)  # [1, N, 1]
    mask_in = token_mask.to(device).view(1, -1, 1)

    # 扩展维度以匹配 PolicyNet 的输入要求
    #  d_glob=1, d_loc=1, d_pos=1
    state = torch.cat([prior, mask_in, mask_in, mask_in], dim=-1)

    # 2. 动作生成 (Action Generation)
    logits, value = policy_net(state)

    # 训练时使用 Bernoulli 采样
    new_mask, log_prob, entropy = policy_net.sample_action(logits, training=True)
    new_mask = new_mask.squeeze(0).squeeze(-1)  # [N]

    # 3. 奖励计算 (Reward Calculation)
    # 计算混合奖励
    # A. 准确率 (MSE or Acc)
    # 这里使用简单的 Hook 评估准确率作为奖励
    acc = calculate_accuracy_reward(model, dataloader, new_mask, device, num_eval_samples)

    # B. 预算惩罚 (Budget Penalty)
    current_count = new_mask.sum().item()
    budget_penalty = -((current_count - target_token_count) / seq_len) ** 2

    reward = acc + budget_penalty

    # 4. 损失计算 (Loss Calculation)
    # A3C / A2C Loss
    # Advantage = R - V(s)
    advantage = reward - value.detach().squeeze()

    critic_loss = (reward - value.squeeze()).pow(2)
    actor_loss = -(log_prob * advantage)

    total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

    # 5. 反向传播与更新
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    optimizer.step()

    return new_mask, {
        'loss': total_loss.item(),
        'reward': reward,
        'acc': acc,
        'kept': current_count
    }


def calculate_accuracy_reward(model, dataloader, mask, device, num_samples=1000):
    """
    辅助函数：计算准确率奖励
    """
    model.eval()
    hook_handle = None

    def hook(module, input, output):
        m = mask.to(output.device).view(1, -1, 1)
        if m.shape[1] == output.shape[1]:
            m[0, 0, 0] = 1.0  # Keep CLS
            return output * m
        return output

    try:
        # 尝试注册到 Embedding Dropout 层
        for name, module in model.named_modules():
            if 'embeddings.dropout' in name:
                hook_handle = module.register_forward_hook(hook)
                break

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if total_samples >= num_samples: break

                if isinstance(batch, list):
                    imgs, lbls = batch[0], batch[1]
                else:
                    imgs, lbls = batch['pixel_values'], batch['labels']

                imgs = imgs.to(device)
                lbls = lbls.to(device)

                outputs = model(pixel_values=imgs)
                preds = outputs.logits.argmax(dim=-1)

                total_correct += (preds == lbls).sum().item()
                total_samples += lbls.size(0)

        return total_correct / total_samples if total_samples > 0 else 0.0

    finally:
        if hook_handle: hook_handle.remove()