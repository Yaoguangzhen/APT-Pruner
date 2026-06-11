# evaluate/imagenet.py
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_imagenet_acc(model, token_mask, dataloader, task_name, device=None):
    """
    评估模型在 ImageNet 数据集上的 top-1 准确率，并应用 Token Mask。

    Args:
        model: 需要评估的 PyTorch 模型。
        token_mask: 1D Tensor [seq_len], 指定保留哪些token。
        dataloader: 提供 ImageNet 测试数据的 DataLoader。
        task_name: 任务名称 (用于日志)。
        device: 运行设备 (可选)。

    Returns:
        float: 模型在数据集上的 top-1 准确率。
    """
    # 1. 设备设置
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # 2. 定义 Hook 以应用 Mask (核心逻辑)
    # 如果没有这个 hook，传入的 token_mask 就不会生效！
    hook_handle = None

    def apply_token_mask_hook(module, input, output):
        if token_mask is None:
            return output

        # 确保 mask 在正确设备上
        mask = token_mask.to(output.device, dtype=output.dtype)

        # 强制保留 CLS Token
        if mask.numel() > 0:
            mask[0] = 1.0

        # 扩展维度 [Seq] -> [1, Seq, 1] 以便广播
        expanded_mask = mask.view(1, -1, 1)

        # 软剪枝：将被 Mask 的 Token 置零
        return output * expanded_mask

    # 3. 注册 Hook (自动查找适合的层)
    # 尝试常见的 Transformer Embedding Dropout 层命名
    target_layer = None
    possible_layer_names = [
        'vit.embeddings.dropout',  # HuggingFace Standard ViT
        'embeddings.dropout',  # Generic / DeiT
        'backbone.embeddings.dropout'  # ViTDet / MaskRCNN
    ]

    for name in possible_layer_names:
        try:
            parts = name.split('.')
            curr = model
            for part in parts:
                curr = getattr(curr, part)
            target_layer = curr
            logger.info(f"[{task_name}] Hook registered on layer: {name}")
            break
        except AttributeError:
            continue

    if target_layer is None:
        logger.warning(f"[{task_name}] Could not find embedding dropout layer. Evaluating WITHOUT mask!")
    else:
        hook_handle = target_layer.register_forward_hook(apply_token_mask_hook)

    try:
        total_correct = 0
        total_samples = 0

        # 4. 评估循环
        for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
            # 兼容不同的数据加载器返回格式
            if isinstance(batch, dict):
                pixel_values = batch.get('pixel_values')
                labels = batch.get('labels')
            elif isinstance(batch, (list, tuple)):
                pixel_values, labels = batch[0], batch[1]
            else:
                logger.warning("Unknown batch format. Skipping.")
                continue

            # 移至 GPU
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if pixel_values is None or labels is None:
                continue

            # 前向传播 (Hook 会在此时被触发，应用 mask)
            outputs = model(pixel_values=pixel_values)

            # 获取预测
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs  # 兼容直接返回 Tensor 的模型

            predictions = logits.argmax(dim=-1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        logger.info(f"[{task_name}] Evaluation finished. Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")

    finally:
        # 5. 清理 Hook
        if hook_handle is not None:
            hook_handle.remove()

    return accuracy