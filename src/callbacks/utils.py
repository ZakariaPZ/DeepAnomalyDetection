import typing as th
import torch
import torchvision
from pytorch_lightning.loggers import WandbLogger


def log_images_wandb(
    logger,
    key: str,
    images: th.List[torch.Tensor],
    captions: th.Optional[th.List[str]] = None,
    **kwargs,
):
    # print("logging images with wandb", key, captions, len(images))
    logger.log_image(
        key=key,
        images=images if isinstance(images, list) else [images],
        caption=captions,
    )


def log_images(
    logger,
    key: str,
    images: th.List[torch.Tensor],
    captions: th.Optional[th.List[str]] = None,
    global_step: th.Optional[int] = None,
    **kwargs,
):
    if isinstance(logger, WandbLogger) or not hasattr(logger.experiment, "add_images"):
        log_images_wandb(logger, key, images, captions)
    else:
        # then assume it is a tensorboard logger
        logger.experiment.add_images(
            key,
            torchvision.utils.make_grid(images, **kwargs).unsqueeze(0),
            global_step=global_step,
        )
