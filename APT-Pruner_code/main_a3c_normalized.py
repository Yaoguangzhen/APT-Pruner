import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Bernoulli
from efficiency.mac import compute_token_pruned_mac
# 假设你已经有了 FactorizedPolicyNet，如果没有可以使用原来的 PolicyNet
from prune.policy_net import PolicyNet
import logging

logger = logging.getLogger(__name__)


def scale_to_comparable_range(fi_term, tp_term, c_term):
    """
    [保留你的改进] 自适应缩放奖励组件
    """
    # ... (保持你原来 main_a3c_normalized.py 中的逻辑) ...
    # 为了简洁，这里省略具体实现细节，实际运行时请确保包含该函数
    # 核心逻辑：确保每个非零项至少达到最大项的 10%
    values = [abs(x.item()) for x in [fi_term, tp_term, c_term]]
    max_val = max(values) if any(values) else 0
    if max_val == 0: return fi_term, tp_term, c_term

    scaled = []
    for x in [fi_term, tp_term, c_term]:
        val = abs(x.item())
        if val > 0 and val < 0.1 * max_val:
            scale = (0.1 * max_val) / val
            scaled.append(x * scale)
        else:
            scaled.append(x)
    return scaled[0], scaled[1], scaled[2]


class A3CPruningWorker(mp.Process):
    def __init__(self, worker_id, args, global_net, optimizer, static_prior, dataset, backbone_model, reward_config):
        super(A3CPruningWorker, self).__init__()
        self.worker_id = worker_id
        self.args = args
        self.global_net = global_net
        self.optimizer = optimizer
        self.static_prior = static_prior  # Fisher Info
        self.dataset = dataset  # 这是一个 Subset
        self.backbone_model = backbone_model  # 每个Worker独立的模型副本
        self.reward_config = reward_config
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    def run(self):
        logger.info(f"Worker {self.worker_id} started on {self.device}")

        # 1. 设置环境
        self.backbone_model.to(self.device)
        self.backbone_model.eval()

        # 2. 初始化本地策略网络 (从全局网络同步)
        # 注意：这里假设输入维度是 seq_len * 2 (Fisher + Mask)
        input_dim = self.static_prior.shape[0] * 2
        local_net = PolicyNet(input_dim).to(self.device)  # 或者 FactorizedPolicyNet

        # 3. 准备数据加载器 (每个Worker自己采样)
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=True, num_workers=0
        )
        data_iter = iter(dataloader)

        step = 0
        current_mask = torch.ones_like(self.static_prior).to(self.device)  # 初始Mask

        while step < self.args.max_steps:
            # --- 同步参数 ---
            local_net.load_state_dict(self.global_net.state_dict())

            # --- 准备状态 (State) ---
            # 你的代码目前使用 [Fisher, Mask] 拼接
            fisher_on_device = self.static_prior.to(self.device)
            state = torch.cat([fisher_on_device, current_mask], dim=0)

            # --- 动作决策 (Action) ---
            keep_probs = local_net(state)  # PolicyNet forward
            dist = Bernoulli(probs=torch.clamp(keep_probs, 1e-6, 1 - 1e-6))

            # 采样新 Mask
            new_mask = dist.sample()
            if new_mask.sum() == 0: new_mask[0] = 1.0  # 保证至少保留CLS

            # 计算 Log Probs (用于 Loss)
            log_prob = dist.log_prob(new_mask).sum()

            # --- 计算奖励 (Reward) ---
            # 1. Fisher Term
            fi_term = torch.tensor(0.0, device=self.device)
            if self.reward_config.use_fi_term:
                # 你的改进：先放大防止下溢，再计算
                fi_term = (fisher_on_device * new_mask).mean()

            # 2. Accuracy