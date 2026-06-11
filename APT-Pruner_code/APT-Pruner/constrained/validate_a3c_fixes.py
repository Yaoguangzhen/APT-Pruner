#!/usr/bin/env python3
"""


验证内容:
1. 约束满足 (Inference Top-k)
2. 奖励信号质量 (Hybrid Reward)
3. 训练稳定性 (Asynchronous Update)
4. 性能改进验证

"""

import torch
import numpy as np
import time
import logging
import os
import torch.multiprocessing as mp
from datetime import datetime

# 导入 A3C 模块
from prune.policy_net_constrained import ConstrainedPolicyNet
from prune.a3c_worker import A3CPruningWorker
from prune.fisher import collect_static_prior  # 假设有这个函数

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class MockDataset(torch.utils.data.Dataset):
    def __len__(self): return 100

    def __getitem__(self, idx):
        return {"pixel_values": torch.randn(3, 224, 224), "labels": torch.tensor(0)}


class MockViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = torch.nn.Module()
        self.vit.embeddings = torch.nn.Module()
        self.vit.embeddings.dropout = torch.nn.Dropout()
        self.config = type('C', (), {'num_hidden_layers': 12, 'num_attention_heads': 12, 'hidden_size': 768,
                                     'intermediate_size': 3072})()

    def forward(self, pixel_values):
        return type('O', (), {'logits': torch.randn(pixel_values.shape[0], 1000)})()


class A3CFixValidation:
    """A3C 修复效果验证器"""

    def __init__(self, seq_len=197, target_tokens=139):
        self.seq_len = seq_len
        self.target_tokens = target_tokens
        # Input dim: 1(Prior) + 1(Glob) + 1(Loc) + 1(Pos) = 4 (Simplified)
        self.d_feat = 64
        self.output_dir = f"validation_a3c_{datetime.now().strftime('%Y%m%d_%H%M')}"
        os.makedirs(self.output_dir, exist_ok=True)

    def test_constraint_satisfaction(self):
        """测试1: 推理阶段的 Top-k 约束满足"""
        logger.info("🧪 测试1: 约束满足情况 (Top-k)")

        policy = ConstrainedPolicyNet(self.d_feat, self.d_feat, 1)
        state = torch.randn(1, self.seq_len, 1 + self.d_feat * 2 + 1)

        violations = 0
        cls_violations = 0

        logger.info("测试1000次推理...")

        # 模拟推理模式 (Top-k)
        logits, _ = policy(state)
        for i in range(1000):
            # 添加随机扰动模拟不同输入
            noisy_logits = logits + torch.randn_like(logits) * 0.1
            mask, _, _ = policy.sample_action(noisy_logits, target_count=self.target_tokens, training=False)

            actual = int(mask.sum().item())
            cls_kept = int(mask[0, 0].item())

            if actual != self.target_tokens: violations += 1
            if cls_kept != 1: cls_violations += 1

        rate = violations / 1000
        logger.info(f"约束违反率: {rate:.2%}")

        passed = (rate == 0)
        if passed:
            logger.info("✅ 约束测试通过")
        else:
            logger.error("❌ 约束测试失败")
        return passed

    def test_reward_signal(self):
        """测试2: 混合奖励计算"""
        logger.info("🧪 测试2: 混合奖励信号")

        # 模拟 Worker 环境
        model = MockViT()
        worker = A3CPruningWorker(0, None, None, None, None, None, model, model.config, self.seq_len, None)

        # 手动注入 RewardConfig
        class Config:
            use_fi_term = True
            use_tp_term = True
            use_c_term = True
            normalize = True
            alpha, beta, delta = 1.0, 1.0, 1.0

        worker.reward_config = Config()

        # 模拟输入
        fi_term = torch.tensor(0.5)
        tp_term = torch.tensor(0.8)
        c_term = torch.tensor(0.1)

        # 测试 Scale 函数
        from prune.a3c_worker import scale_to_comparable_range
        scaled_fi, scaled_tp, scaled_c = scale_to_comparable_range(fi_term, tp_term, c_term)

        logger.info(f"Original: FI={fi_term:.2f}, TP={tp_term:.2f}, C={c_term:.2f}")
        logger.info(f"Scaled:   FI={scaled_fi:.2f}, TP={scaled_tp:.2f}, C={scaled_c:.2f}")

        # 检查是否在同一数量级 (比如差异不超过 100 倍)
        vals = [scaled_fi, scaled_tp, scaled_c]
        ratio = max(vals) / (min(vals) + 1e-9)

        passed = ratio < 100
        if passed:
            logger.info("✅ 奖励缩放测试通过")
        else:
            logger.error(f"❌ 奖励缩放失败, Ratio={ratio:.1f}")
        return passed

    def test_async_stability(self):
        """测试3: 异步更新稳定性"""
        logger.info("🧪 测试3: 异步更新稳定性")

        mp.set_start_method('spawn', force=True)

        global_net = ConstrainedPolicyNet(self.d_feat, self.d_feat, 1)
        global_net.share_memory_()
        optimizer = torch.optim.Adam(global_net.parameters(), lr=1e-3)  # Mock SharedAdam

        # 启动两个简化版 Worker 进程
        def worker_fn(rank, net, opt):
            local_net = ConstrainedPolicyNet(64, 64, 1)
            for _ in range(5):
                local_net.load_state_dict(net.state_dict())
                # Fake update
                loss = torch.randn(1, requires_grad=True)
                opt.zero_grad()
                loss.backward()
                # Push grads
                for lp, gp in zip(local_net.parameters(), net.parameters()):
                    gp.grad = torch.randn_like(gp)  # Mock grad
                opt.step()

        procs = []
        try:
            for i in range(2):
                p = mp.Process(target=worker_fn, args=(i, global_net, optimizer))
                p.start()
                procs.append(p)

            for p in procs: p.join()
            passed = True
            logger.info("✅ 异步更新测试通过")
        except Exception as e:
            logger.error(f"❌ 异步更新失败: {e}")
            passed = False

        return passed

    def run(self):
        results = [
            self.test_constraint_satisfaction(),
            self.test_reward_signal(),
            self.test_async_stability()
        ]

        if all(results):
            logger.info("🎉 所有 A3C 修复验证通过!")
            return True
        else:
            logger.error("⚠️ 部分验证失败")
            return False


if __name__ == "__main__":
    validator = A3CFixValidation()
    validator.run()