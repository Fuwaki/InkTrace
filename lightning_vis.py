"""
PyTorch Lightning 可视化回调 (Online Adapter)

职责：
- 作为 PyTorch Lightning 的 Callback 插件
- 在验证周期结束时自动生成对比图
- 将图片写入 TensorBoard

依赖：
- pytorch_lightning
- vis_core (核心渲染层)
"""


import pytorch_lightning as pl
import torch

from vis_core import create_grid_image, compute_metrics


class VisualizationCallback(pl.Callback):
    """
    训练过程中的自动可视化回调

    在每个 validation epoch 结束时：
    1. 从验证 DataLoader 获取一个 batch
    2. 进行模型推理
    3. 调用核心渲染层生成对比图
    4. 写入 TensorBoard

    使用方法：
        trainer = pl.Trainer(
            callbacks=[VisualizationCallback(num_samples=4)]
        )
    """

    def __init__(
        self,
        num_samples: int = 4,
        log_metrics: bool = True,
        prefix: str = "Validation",
    ):
        """
        Args:
            num_samples: 每次可视化的样本数量
            log_metrics: 是否记录评估指标到 TensorBoard
            prefix: TensorBoard 标签前缀
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_metrics = log_metrics
        self.prefix = prefix

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """
        Validation epoch 结束时的回调

        注意：这里使用 val_dataloaders 而不是 train_dataloaders，
        因为验证数据更稳定，且可以在训练过程中监控模型性能
        """
        # 获取验证 DataLoader
        val_dataloaders = trainer.val_dataloaders

        if val_dataloaders is None:
            return

        # 处理两种情况：DataLoader 对象或 DataLoader 列表
        if isinstance(val_dataloaders, list):
            if len(val_dataloaders) == 0:
                return
            dataloader = val_dataloaders[0]
        else:
            dataloader = val_dataloaders

        try:
            # 获取一个 batch
            batch = next(iter(dataloader))
        except StopIteration:
            # DataLoader 为空
            return

        imgs, targets = batch

        # 限制样本数
        imgs = imgs[: self.num_samples]
        targets = {k: v[: self.num_samples] for k, v in targets.items()}

        # 移动到模型所在设备
        device = pl_module.device
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        # 模型推理
        pl_module.eval()
        with torch.no_grad():
            outputs = pl_module(imgs)

        # 恢复训练模式
        pl_module.train()

        # 生成可视化网格
        try:
            grid = create_grid_image(
                imgs,
                outputs,
                targets,
                num_samples=self.num_samples,
            )
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            return

        # 转换为 Tensor (HWC -> CHW, [0, 255] -> [0, 1])
        grid_tensor = torch.from_numpy(grid).permute(2, 0, 1).float() / 255.0

        # 写入 TensorBoard
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            logger = trainer.logger.experiment

            # 兼容多种 logger
            if hasattr(logger, "add_image"):
                # TensorBoard
                logger.add_image(
                    f"{self.prefix}/Visualization",
                    grid_tensor,
                    trainer.global_step,
                )
            elif hasattr(logger, "log_image"):
                # W&B 或其他 logger
                logger.log_image(
                    f"{self.prefix}/Visualization",
                    [grid_tensor],
                    step=trainer.global_step,
                )

        # 计算并记录指标
        if self.log_metrics:
            try:
                metrics = compute_metrics(outputs, targets)
                for name, value in metrics.items():
                    pl_module.log(
                        f"{self.prefix}/{name}",
                        value,
                        prog_bar=False,
                        logger=True,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
            except Exception as e:
                print(f"Warning: Metric computation failed: {e}")


class TrainingVisualizationCallback(pl.Callback):
    """
    训练过程中的可视化回调 (使用训练数据)

    与 VisualizationCallback 的区别：
    - 使用训练数据而不是验证数据
    - 可以更频繁地调用 (例如每 N 个 batch)
    - 用于监控训练过程中的即时预测

    使用方法：
        trainer = pl.Trainer(
            callbacks=[
                TrainingVisualizationCallback(
                    num_batches=10,  # 每 10 个 batch 可视化一次
                    num_samples=4,
                )
            ]
        )
    """

    def __init__(
        self,
        num_batches: int = 10,
        num_samples: int = 4,
        prefix: str = "Train",
    ):
        """
        Args:
            num_batches: 每隔多少个 batch 可视化一次
            num_samples: 每次可视化的样本数量
            prefix: TensorBoard 标签前缀
        """
        super().__init__()
        self.num_batches = num_batches
        self.num_samples = num_samples
        self.prefix = prefix
        self.batch_count = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """每个训练 batch 结束时的回调"""
        self.batch_count += 1

        # 按频率触发
        if self.batch_count % self.num_batches != 0:
            return

        imgs, targets = batch

        # 限制样本数
        imgs = imgs[: self.num_samples]
        targets = {k: v[: self.num_samples] for k, v in targets.items()}

        # 移动到模型所在设备
        device = pl_module.device
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        # 模型推理
        with torch.no_grad():
            pred_outputs = pl_module(imgs)

        # 生成可视化
        try:
            grid = create_grid_image(
                imgs,
                pred_outputs,
                targets,
                num_samples=self.num_samples,
            )
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            return

        # 转换为 Tensor
        grid_tensor = torch.from_numpy(grid).permute(2, 0, 1).float() / 255.0

        # 写入 TensorBoard
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            logger = trainer.logger.experiment

            if hasattr(logger, "add_image"):
                logger.add_image(
                    f"{self.prefix}/Visualization",
                    grid_tensor,
                    trainer.global_step,
                )
