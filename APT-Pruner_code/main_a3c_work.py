import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributions import Bernoulli
from efficiency.mac import compute_token_pruned_mac
from prune.policy_net import PolicyNet, CriticNet
import logging
import traceback
import numpy as np

logger = logging.getLogger(__name__)


# --- 辅助函数：自适应缩放奖励 ---
def scale_to_comparable_range(fi_term, tp_term, c_term):
    """确保奖励组件在数量级上可比"""
    device = fi_term.device
    fi_val, tp_val, c_val = fi_term.item(), tp_term.item(), c_term.item()

    if fi_val == 0 and tp_val == 0 and c_val == 0:
        return fi_term, tp_term, c_term

    values = [abs(fi_val), abs(tp_val), abs(c_val)]
    positive_values = [v for v in values if v > 0]
    if not positive_values:
        return fi_term, tp_term, c_term

    max_val = max(positive_values)
    scaling_factors = []

    # 目标：让每个非零项至少达到最大值的 10%
    for val in [fi_val, tp_val, c_val]:
        if val == 0:
            scaling_factors.append(1.0)
        else:
            scale = max(1.0, 0.1 * max_val / abs(val))
            scaling_factors.append(scale)

    return (fi_term * scaling_factors[0],
            tp_term * scaling_factors[1],
            c_term * scaling_factors[2])


class A3CPruningWorker(mp.Process):
    def __init__(self, worker_id, args, global_policy, global_critic, optimizer,
                 static_prior, dataset, backbone_model, config, seq_len, reward_config):
        super(A3CPruningWorker, self).__init__()
        self.worker_id = worker_id
        self.args = args
        self.global_policy = global_policy
        self.global_critic = global_critic
        self.optimizer = optimizer

        # 共享的静态先验 (Fisher Info)
        self.static_prior = static_prior

        # 每个Worker独立的数据集子集和模型副本
        self.dataset = dataset
        self.backbone_model = backbone_model

        # 配置
        self.config = config
        self.seq_len = seq_len
        self.reward_config = reward_config

        # 设备 (建议根据显存情况分配，显存不够则使用CPU)
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    def run(self):
        logger.info(f"Worker {self.worker_id} 启动，使用设备: {self.device}")

        # 1. 设置本地环境
        self.backbone_model.to(self.device)
        self.backbone_model.eval()

        # 2. 初始化本地网络 (结构与全局网络一致)
        input_dim = self.seq_len * 2
        local_policy = PolicyNet(input_dim).to(self.device)
        local_critic = CriticNet(input_dim).to(self.device)

        # 3. 准备数据加载器
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=True, num_workers=0
        )
        data_iter = iter(dataloader)

        step = 0
        current_mask = torch.ones_like(self.static_prior).to(self.device)

        try:
            while step < self.args.max_steps:
                # --- A. 同步全局参数 ---
                local_policy.load_state_dict(self.global_policy.state_dict())
                local_critic.load_state_dict(self.global_critic.state_dict())

                # --- B. 构建状态 (State) ---
                fisher_on_device = self.static_prior.to(self.device)
                mask_on_device = current_mask.to(self.device)

                # 状态 = [Fisher信息, 当前Mask]
                state = torch.cat([fisher_on_device, mask_on_device], dim=0)

                # --- C. 动作决策 (Action) ---
                probs = local_policy(state)
                dist = Bernoulli(probs=torch.clamp(probs, 1e-6, 1.0 - 1e-6))

                new_mask = dist.sample()
                if new_mask.sum() == 0: new_mask[0] = 1.0  # 保证至少保留CLS

                # --- D. 计算奖励 (Reward) ---
                # 1. Fisher项
                fi_term = torch.tensor(0.0, device=self.device)
                if self.reward_config.use_fi_term:
                    # 数值稳定性处理：放大计算再缩小
                    scaled_fisher = fisher_on_device * 1e6
                    fi_term = (scaled_fisher * new_mask).mean() / 1e6

                # 2. 准确率项 (耗时操作，现在并行化了)
                tp_term = torch.tensor(0.0, device=self.device)
                if self.reward_config.use_tp_term:
                    acc = self._evaluate_accuracy(new_mask, data_iter, dataloader)
                    tp_term = torch.tensor(acc, device=self.device)

                # 3. MAC约束项
                c_term = torch.tensor(0.0, device=self.device)
                if self.reward_config.use_c_term:
                    kept = int(new_mask.sum().item())
                    p_mac, o_mac = compute_token_pruned_mac(self.config, self.seq_len, kept)
                    ratio = (o_mac - p_mac) / o_mac if o_mac > 0 else 0
                    c_term = torch.tensor(ratio, device=self.device)

                # 归一化与加权
                if self.reward_config.normalize:
                    fi_term, tp_term, c_term = scale_to_comparable_range(fi_term, tp_term, c_term)

                final_reward = (self.reward_config.alpha * fi_term +
                                self.reward_config.beta * tp_term +
                                self.reward_config.delta * c_term)

                # --- E. 计算损失 (Loss) ---
                # Critic 损失 (MSE)
                value_pred = local_critic(state)
                # 在单步设置下，目标价值就是当前奖励 (Advantage = Reward - Value)
                # 如果是多步，需要计算TD-Target
                advantage = final_reward - value_pred.detach()
                critic_loss = (final_reward - value_pred).pow(2)

                # Actor 损失 (PPO/REINFORCE with Baseline)
                log_prob = dist.log_prob(new_mask).sum()
                actor_loss = -(log_prob * advantage)

                # 熵正则化
                entropy_loss = -self.args.ent_coef * dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                # --- F. 反向传播与异步更新 ---
                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(local_policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(local_critic.parameters(), 0.5)

                # 将本地梯度复制到全局网络 (关键步骤)
                for lp, gp in zip(local_policy.parameters(), self.global_policy.parameters()):
                    if gp.grad is None:
                        gp.grad = lp.grad.detach()
                    else:
                        gp.grad += lp.grad.detach()  # 累积梯度 (或直接赋值，取决于实现)

                for lc, gc in zip(local_critic.parameters(), self.global_critic.parameters()):
                    if gc.grad is None:
                        gc.grad = lc.grad.detach()
                    else:
                        gc.grad += lc.grad.detach()

                # 全局优化器更新
                self.optimizer.step()

                # 更新状态
                current_mask = new_mask.detach()
                step += 1

                if step % 10 == 0:
                    logger.info(
                        f"Worker {self.worker_id} Step {step}: Reward={final_reward.item():.4f} (Acc={tp_term.item():.2f})")

        except Exception as e:
            logger.error(f"Worker {self.worker_id} 异常退出: {e}")
            logger.error(traceback.format_exc())

    def _evaluate_accuracy(self, mask, data_iter, dataloader):
        """快速评估当前Mask的准确率 (使用少量Batch)"""

        # 定义Hook
        def mask_hook(module, input, output):
            m = mask.to(output.device)
            return output * m.unsqueeze(0).unsqueeze(-1)

        hook = self.backbone_model.vit.embeddings.dropout.register_forward_hook(mask_hook)

        total_acc = 0
        batches = 0
        try:
            # 只评估几个Batch以加快速度
            for _ in range(5):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)  # 重置迭代器
                    batch = next(data_iter)

                if isinstance(batch, (list, tuple)):
                    imgs, lbl