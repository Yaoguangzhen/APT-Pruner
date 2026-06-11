import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from dataset.dataset import get_image_dataset

# 设置日志
logger = logging.getLogger(__name__)

# 处理 Autocast 导入，确保兼容性
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
else:
    # 如果没有CUDA，定义一个空的上下文管理器
    class autocast:
        def __init__(self, enabled=False, dtype=None):
            self.enabled = enabled

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


@torch.no_grad()
def test_accuracy(model, token_mask, image_processor, task_name, data_dir="./data", img_size=224, batch_size=32,
                  device=None, num_workers=4):
    """
    在验证集上评估模型准确率，并应用 Token Mask。

    Args:
        model: 待评估模型
        token_mask: 1D Tensor [seq_len], 指定保留哪些token
        image_processor: 图片预处理器
        task_name: 任务名称 (用于日志)
        data_dir: 数据集路径
        img_size: 图片尺寸
        batch_size: 批次大小
        device: 运行设备
        num_workers: DataLoader进程数
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    # --- 1. 定义 Hook 以应用 Mask ---
    hook_handle = None

    def apply_token_mask_hook(module, input, output):
        # token_mask 形状通常为 [Seq_len]
        # output 形状通常为 [Batch, Seq_len, Hidden_dim]

        if token_mask is None:
            return output

        # 确保 mask 在正确的设备上
        mask = token_mask.to(output.device, dtype=output.dtype)

        # 强制保留 CLS Token (索引 0)
        if mask.numel() > 0:
            mask[0] = 1.0

            # 扩展维度以支持广播: [Seq] -> [1, Seq, 1]
        expanded_mask = mask.view(1, -1, 1)

        # 应用 Mask (软剪枝: 将被剪枝的 token 置零)
        modified_output = output * expanded_mask
        return modified_output

    # --- 2. 注册 Hook (自动查找适合的层) ---
    target_layer = None
    # 尝试常见的 Transformer Embedding Dropout 层命名
    possible_layer_names = [
        'vit.embeddings.dropout',  # HuggingFace Standard ViT
        'embeddings.dropout',  # Generic / DeiT
        'backbone.embeddings.dropout'  # ViTDet / MaskRCNN
    ]

    for name in possible_layer_names:
        try:
            # 递归查找属性
            parts = name.split('.')
            curr = model
            for part in parts:
                curr = getattr(curr, part)
            target_layer = curr
            logger.info(f"Hook registered on layer: {name}")
            break
        except AttributeError:
            continue

    if target_layer is None:
        logger.warning("Could not find embedding dropout layer to apply mask. Evaluating WITHOUT mask!")
    else:
        hook_handle = target_layer.register_forward_hook(apply_token_mask_hook)

    try:
        # --- 3. 准备数据集 ---
        test_dataset = get_image_dataset(
            data_dir,
            'val',
            image_processor=image_processor,
            max_size=img_size
        )

        # 通用 Collate 函数，兼容 dict 和 tuple 返回格式
        def robust_collate_fn(batch):
            if isinstance(batch[0], dict):
                # 如果 Dataset 已经返回字典 (如 HF Dataset)
                return {
                    'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
                    'labels': torch.tensor([b['labels'] for b in batch])
                }
            else:
                # 如果 Dataset 返回 (img, label) 元组 (如 ImageFolder)
                images, labels = zip(*batch)
                return {
                    'pixel_values': torch.stack(images),
                    'labels': torch.tensor(labels)
                }

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,  # 增加并发加载
            collate_fn=robust_collate_fn
        )

        # --- 4. 评估循环 ---
        total_correct = 0
        total_samples = 0

        for batch in tqdm(test_dataloader, desc=f"Eval {task_name}"):
            pixel_values = batch.get('pixel_values').to(device)
            labels = batch.get('labels').to(device)

            if pixel_values is None or labels is None:
                continue

            # 使用 autocast(enabled=False) 禁用混合精度
            # 剪枝/Masking 在 FP16 下有时会导致数值不稳定，这里保持你原有的逻辑
            with torch.no_grad():
                with autocast(enabled=False):
                    outputs = model(pixel_values=pixel_values)
                    logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples if total_samples > 0 else 0

    except Exception as e:
        logger.error(f"Error during test_accuracy: {e}")
        # traceback.print_exc()
        accuracy = 0.0

    finally:
        # --- 5. 清理 Hook ---
        if hook_handle is not None:
            hook_handle.remove()

    return accuracy