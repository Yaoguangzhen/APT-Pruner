import torch
import numpy as np


def collect_static_prior(model, calibration_loader, num_batches=32):
    """
    对应 Sec 3.1: Rapid Static Prior Extraction
    计算 Global Importance Matrix I (Effective Fisher Information)

    Returns:
        importance_matrix: [N_patches] (Avg squared gradient)
        stats: dict containing 'mu' and 'sigma' for Z-score normalization
    """
    model.eval()
    device = next(model.parameters()).device

    # Virtual Gate 梯度累积器
    # 注意：这里的 seq_len 需要根据模型配置自动获取，这里做通用处理
    raw_importances = []

    target_layer_output = None

    # Virtual Gate Hook: 模拟 g * e_{t-1}
    # 对 LayerNorm 或 Embedding 输出求导等价于对系数 g=1 求导
    def hook_fn(module, input, output):
        nonlocal target_layer_output
        target_layer_output = output
        target_layer_output.retain_grad()

    # 尝试挂载到 Patch Embeddings 之后
    try:
        hook_handle = model.vit.embeddings.dropout.register_forward_hook(hook_fn)
    except AttributeError:
        # Fallback for other architectures (e.g., DeiT)
        print("Warning: Standard ViT path not found, trying blocks[0] input...")
        hook_handle = model.blocks[0].register_forward_hook(hook_fn)

    print(f"Collecting Static Prior from {num_batches} batches...")

    batch_count = 0
    for batch in calibration_loader:
        if batch_count >= num_batches: break

        # 适配不同的 Dataloader 输出
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0].to(device), batch[1].to(device)
        elif isinstance(batch, dict):
            inputs = batch['pixel_values'].to(device)
            targets = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else F.cross_entropy(outputs, targets)

        loss.backward()

        if target_layer_output is not None and target_layer_output.grad is not None:
            # Eq. 4 & 5: Expected Squared Gradient
            # grad shape: [B, N, D]
            # 计算 L2 norm over channel dim: [B, N]
            grads = target_layer_output.grad.detach()
            # EFI metric: Squared L2 Norm of gradient
            importance = grads.norm(p=2, dim=-1).pow(2)
            raw_importances.append(importance)

        batch_count += 1

    hook_handle.remove()

    if not raw_importances:
        raise RuntimeError("No gradients collected for Static Prior.")

    # Stack and Average: [Total_B, N] -> Mean -> [N]
    all_importances = torch.cat(raw_importances, dim=0)
    global_importance_I = all_importances.mean(dim=0)  #

    # Pre-compute Statistics for Z-score normalization (Eq. 7)
    # mu_t and sigma_t (treating the whole sequence as the distribution)
    mu = global_importance_I.mean()
    sigma = global_importance_I.std()

    return global_importance_I, {'mu': mu, 'sigma': sigma}