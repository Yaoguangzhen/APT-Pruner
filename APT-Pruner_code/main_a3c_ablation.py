import argparse
import logging
import os
import time
import copy
import torch
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor, set_seed

from dataset.dataset import get_image_dataset
from prune.a3c_worker import A3CPruningWorker  # 导入新的 Worker 类
from prune.fisher import collect_static_prior  # 导入 Stage 1 函数
from prune.policy_net import FactorizedPolicyNet
from utils.utils import SharedAdam  # 假设你有这个工具，或者使用 torch.optim.Adam

logger = logging.getLogger(__name__)


# --- 保留你的配置类 ---
class RewardConfig:
    """配置PPO奖励函数组件的类"""

    def __init__(self, use_fi_term=True, use_tp_term=True, use_c_term=True,
                 alpha=1.0, beta=0.5, delta=0.5):
        self.use_fi_term = use_fi_term
        self.use_tp_term = use_tp_term
        self.use_c_term = use_c_term
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def __str__(self):
        components = []
        if self.use_fi_term: components.append(f"fi_term(α={self.alpha})")
        if self.use_tp_term: components.append(f"tp_term(β={self.beta})")
        if self.use_c_term: components.append(f"c_term(δ={self.delta})")
        return " + ".join(components) if components else "No rewards"


EXPERIMENT_CONFIGS = {
    "original": RewardConfig(True, True, True, 1.0, 0.5, 0.5),
    "v1": RewardConfig(True, True, False, 1.0, 0.5, 0.0),
    "v2": RewardConfig(False, True, True, 0.0, 0.5, 0.5),
    "v3": RewardConfig(False, True, False, 0.0, 1.0, 0.0),
    "v4": RewardConfig(False, True, False, 0.0, 2.0, 0.0),
}

# --- 参数解析 ---
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True, choices=["imagenet", "custom_task"])
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--constraint", type=float, required=True)
parser.add_argument("--seed", type=int, default=0)
# A3C 特定参数
parser.add_argument("--num_workers", type=int, default=4, help="Number of A3C workers")
parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per worker")
# 消融实验参数
parser.add_argument("--experiment", type=str, choices=list(EXPERIMENT_CONFIGS.keys()), required=True)
parser.add_argument("--ablation_name", type=str, default="reward_ablation_study")

# 网络维度参数 (根据你的 PolicyNet 调整)
parser.add_argument("--d_glob", type=int, default=64)
parser.add_argument("--d_loc", type=int, default=384)  # ViT-Base hidden size / 2 or similar
parser.add_argument("--d_pos", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=32)


def main():
    start = time.time()
    args = parser.parse_args()

    # 获取奖励配置
    reward_config = EXPERIMENT_CONFIGS[args.experiment]

    # 设置输出目录
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs", args.model_name, args.task_name, "a3c",
            str(args.constraint), args.ablation_name, f"{args.experiment}_{timestamp}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                      logging.FileHandler(os.path.join(args.output_dir, "log.txt"))])
    logger.info(f"Running A3C Ablation: {args.experiment}")
    logger.info(f"Reward Config: {reward_config}")

    # 1. 设置环境
    mp.set_start_method('spawn', force=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # 2. 加载模型与数据
    logger.info("Loading model and data...")
    config = AutoConfig.from_pretrained(args.ckpt_dir)
    model = AutoModelForImageClassification.from_pretrained(args.ckpt_dir, config=config, ignore_mismatched_sizes=True)
    image_processor = AutoImageProcessor.from_pretrained(args.ckpt_dir, use_fast=True)

    dataset = get_image_dataset(args.data_dir, 'train', image_processor=image_processor, max_size=224)
    # 简单切分一个 calibration set 用于 Stage 1
    calib_len = 1000
    calib_dataset = torch.utils.data.Subset(dataset, range(calib_len))
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=32, shuffle=False)

    # 3. Stage 1: Static Prior Extraction (EFI)
    logger.info(">>> Stage 1: Extracting Static Prior...")
    model.to(device)
    importance_matrix, prior_stats = collect_static_prior(model, calib_loader, num_batches=20)
    # 将 importance_matrix 移回 CPU 共享内存，以免占用 Worker 显存
    importance_matrix = importance_matrix.cpu().share_memory_()

    # 4. Stage 2: A3C Training
    logger.info(">>> Stage 2: Starting A3C Workers...")

    # 初始化全局策略网络
    global_net = FactorizedPolicyNet(args.d_glob, args.d_loc, args.d_pos).to(device)
    global_net.share_memory()  # 关键

    optimizer = SharedAdam(global_net.parameters(), lr=1e-4)  # 使用 SharedAdam

    workers = []
    for i in range(args.num_workers):
        # 深度拷贝模型给 Worker (每个 Worker 拥有独立的模型副本用于环境交互)
        # 注意：这会消耗显存。显存不足时需减少 num_workers 或使用模型共享策略
        worker_model = copy.deepcopy(model).cpu()

        w = A3CPruningWorker(
            worker_id=i,
            args=args,
            global_net=global_net,
            optimizer=optimizer,
            static_prior=importance_matrix,
            prior_stats=prior_stats,
            dataset=dataset,
            backbone_model=worker_model,
            reward_config=reward_config  # <--- 传入消融实验配置
        )
        w.start()
        workers.append(w)
        logger.info(f"Worker {i} started.")

    for w in workers:
        w.join()

    logger.info("A3C Training Finished.")

    # 保存训练好的策略
    torch.save(global_net.state_dict(), os.path.join(args.output_dir, "policy_net.pth"))

    # 5. Final Evaluation (使用 Global Net 进行确定性推理)
    # ... (添加推理代码，类似于之前的 Top-K 选择) ...


if __name__ == "__main__":
    main()