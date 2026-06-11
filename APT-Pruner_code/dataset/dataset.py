import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, DefaultDataCollator

TASKS = ["imagenet", "custom_task"]


class ImageFolderDataset(Dataset):
    """
    包装 ImageFolder，使用 HuggingFace image_processor 进行预处理。
    """

    def __init__(self, root_dir, image_processor):
        self.dataset = ImageFolder(root=root_dir)
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # image 是 PIL Image
        # image_processor 处理后返回 dict {'pixel_values': tensor}
        # 确保图片转为 RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.image_processor(images=image, return_tensors="pt")

        # image_processor 返回的 pixel_values 通常是 [1, C, H, W]，需要 squeeze 掉 batch 维
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label)
        }


def get_image_dataset(data_dir, split_name, image_processor, max_size=224):
    """
    获取指定 split (train/val) 的数据集
    """
    split_dir = os.path.join(data_dir, split_name)

    if not os.path.exists(split_dir):
        # 尝试一些常见的变体，例如 'validation' 代替 'val'
        if split_name == 'val' and os.path.exists(os.path.join(data_dir, 'validation')):
            split_dir = os.path.join(data_dir, 'validation')
        else:
            raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

    # 使用自定义的 Dataset 类，它会在 getitem 时调用 image_processor
    dataset = ImageFolderDataset(root_dir=split_dir, image_processor=image_processor)
    return dataset


def get_image_dataloader(task_name, image_processor, training, batch_size=32, max_size=224, data_dir="./data",
                         num_workers=4):
    split = 'train' if training else 'val'
    dataset = get_image_dataset(data_dir, split, image_processor, max_size=max_size)

    # collate_fn: 将 list of dicts 转换为 dict of tensors (batch)
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers  # 建议设置 num_workers > 0 加速加载
    )
    return dataloader


def get_datasets_and_dataloaders(task_name="imagenet", model_name="google/vit-base-patch16-224-in21k", batch_size=32,
                                 max_size=224, data_dir="./data"):
    # 加载对应模型的 Image Processor (包含 Resize, Normalize 配置)
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        # 如果模型没有 config (例如本地路径不完整)，回退到默认
        from transformers import ViTImageProcessor
        image_processor = ViTImageProcessor(size={"height": max_size, "width": max_size})
        print("Warning: Could not load AutoImageProcessor, using default ViTImageProcessor.")

    # 训练集加载器
    train_loader = get_image_dataloader(
        task_name,
        image_processor,
        training=True,
        batch_size=batch_size,
        max_size=max_size,
        data_dir=data_dir
    )

    # 测试集加载器
    test_loader = get_image_dataloader(
        task_name,
        image_processor,
        training=False,
        batch_size=batch_size,
        max_size=max_size,
        data_dir=data_dir
    )

    return train_loader, test_loader