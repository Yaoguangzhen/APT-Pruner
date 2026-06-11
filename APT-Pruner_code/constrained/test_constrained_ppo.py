#!/usr/bin/env python3
"""
约束感知 A3C 的测试脚本 (Test Script for Constraint-Aware A3C)

测试内容:
1. Factorized Policy 的非对称采样 (Asymmetric Sampling)
2. 推理模式下的严格约束满足 (Strict Constraint Satisfaction)
3. A3C Worker 的单步更新逻辑
4. 边界情况测试

"""

import torch
import torch.nn as nn
import time
import numpy as np
import logging
from prune.policy_net_constrained import ConstrainedPolicyNet
from prune.a3c_worker import A3CPruningWorker
from dataset.dataset import ImageFolderDataset  # Mock dataset needed

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockArgs:
    def __init__(self):
        self.gpu = 0
        self.seed = 42
        self.max_steps = 10
        self.constraint = 0.7


class MockDataset(torch.utils.data.Dataset):
    def __len__(self): return 100

    def __getitem__(self, idx):
        # Return dummy pixel_values and labels
        return {"pixel_values": torch.randn(3, 224, 224), "labels": torch.tensor(0)}


class MockViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = nn.Module()
        self.vit.embeddings = nn.Module()
        self.vit.embeddings.dropout = nn.Dropout()
        self.config = type('Config', (), {'num_hidden_layers': 12, 'num_attention_heads': 12, 'hidden_size': 768,
                                          'intermediate_size': 3072})()

    def forward(self, pixel_values):
        # Dummy output
        batch_size = pixel_values.shape[0]
        return type('Output', (), {'logits': torch.randn(batch_size, 1000)})()


def test_factorized_policy():
    """测试 Factorized Policy 的非对称采样"""
    print("\n🧪 测试1: 策略网络非对称采样")
    print("-" * 40)

    B, N = 2, 197
    d_glob, d_loc, d_pos = 64, 64, 32
    target_k = 139

    policy = ConstrainedPolicyNet(d_glob, d_loc, d_pos)

    # 模拟输入状态
    input_dim = 1 + d_glob + d_loc + d_pos
    state = torch.randn(B, N, input_dim)

    logits, value = policy(state)

    # 1. 测试训练模式 (Bernoulli)
    mask_train, _, _ = policy.sample_action(logits, training=True)
    count_train = mask_train[0].sum().item()
    print(f"训练模式采样数 (随机): {count_train}")

    # 2. 测试推理模式 (Top-k)
    mask_eval, _, _ = policy.sample_action(logits, target_count=target_k, training=False)
    count_eval = mask_eval[0].sum().item()
    print(f"推理模式采样数 (Top-{target_k}): {count_eval}")

    # 验证 CLS 保护
    cls_selected = mask_eval[0, 0].item() == 1.0
    print(f"CLS Token 保留: {cls_selected}")

    if count_eval == target_k and cls_selected:
        print("✅ 策略网络测试通过!")
        return True
    else:
        print(f"❌ 策略网络测试失败: 目标={target_k}, 实际={count_eval}, CLS={cls_selected}")
        return False


def test_a3c_worker_step():
    """测试 A3C Worker 的单步执行"""
    print("\n🧪 测试2: A3C Worker 单步执行")
    print("-" * 40)

    args = MockArgs()
    global_net = ConstrainedPolicyNet(1, 1, 1)
    optimizer = torch.optim.Adam(global_net.parameters())  # Mock shared optimizer
    fisher_info = torch.randn(197)
    dataset = MockDataset()
    model = MockViT()

    class MockConfig:
        use_fi_term = True
        use_tp_term = True
        use_c_term = True
        normalize = True
        alpha, beta, delta = 1.0, 1.0, 1.0

    worker = A3CPruningWorker(
        worker_id=0,
        args=args,
        global_net=global_net,
        optimizer=optimizer,
        static_prior=fisher_info,
        dataset=dataset,
        backbone_model=model,
        config=model.config,
        seq_len=197,
        reward_config=MockConfig()
    )

    # 强制运行一小段逻辑 (模拟 run 方法中的循环体)
    try:
        worker.backbone_model.to(worker.device)
        local_net = ConstrainedPolicyNet(1, 1, 1).to(worker.device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        data_iter = iter(dataloader)

        # 模拟一步交互
        current_mask = torch.ones(197, device=worker.device)
        prior = fisher_info.to(worker.device).view(1, -1, 1)
        mask_in = current_mask.view(1, -1, 1)
        state = torch.cat([prior, mask_in, mask_in, mask_in], dim=-1)

        logits, value = local_net(state)
        new_mask, log_prob, entropy = local_net.sample_action(logits, training=True)

        # 验证梯度回传
        reward = torch.tensor(1.0, device=worker.device)
        loss = (reward - value.mean()).pow(2)

        optimizer.zero_grad()
        loss.backward()

        # 检查是否有梯度
        has_grad = False
        for param in local_net.parameters():
            if param.grad is not None:
                has_grad = True
                break

        if has_grad:
            print("✅ A3C Worker 梯度计算正常")
            return True
        else:
            print("❌ A3C Worker 无梯度!")
            return False

    except Exception as e:
        print(f"❌ A3C Worker 执行异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """测试边界情况"""
    print("\n🧪 测试3: 边界情况 (极少/极多 Token)")
    print("-" * 40)

    policy = ConstrainedPolicyNet(1, 1, 1)
    logits = torch.randn(1, 197, 1)

    passed = True
    for k in [1, 10, 100, 196, 197]:
        mask, _, _ = policy.sample_action(logits, target_count=k, training=False)
        actual = int(mask.sum().item())
        if actual != k:
            print(f"❌ 失败: Target={k}, Actual={actual}")
            passed = False
        else:
            print(f"✅ 通过: Target={k}")

    return passed


def main():
    print("🚀 开始 A3C 单元测试")
    print("=" * 50)

    tests = [
        test_factorized_policy,
        test_a3c_worker_step,
        test_edge_cases
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    if all(results):
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试失败!")


if __name__ == "__main__":
    main()