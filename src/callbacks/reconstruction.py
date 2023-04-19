import typing as th
import lightning.pytorch as pl
import torch
from .utils import log_images


class ReconstructionCallback(pl.Callback):
    def __init__(
        self,
        best: bool = True,
        worst: bool = True,
    ):
        self.best, self.worst = best, worst

        super().__init__()

        assert best or worst, "At least one of best or worst must be True"

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: th.Optional["STEP_OUTPUT"],  # type: ignore
        batch: th.Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, y = batch
        # get the reconstruction loss for each sample in the batch
        reconstruction_loss = torch.nn.functional.mse_loss(x, pl_module(x), reduction="none").mean(dim=(1, 2, 3))

        # get the indices of the best and worst reconstruction
        best_idx = reconstruction_loss.argsort(descending=False)[0]
        worst_idx = reconstruction_loss.argsort(descending=True)[0]

        best_input = x[best_idx]
        worst_input = x[worst_idx]

        # get the best and worst reconstruction
        best_reconstruction = pl_module(best_input.unsqueeze(0)).squeeze(0)
        worst_reconstruction = pl_module(worst_input.unsqueeze(0)).squeeze(0)

        # log the best and worst reconstruction
        if self.best:
            log_images(
                logger=trainer.logger,
                key="best_reconstruction",
                images=[best_input, best_reconstruction, best_input - best_reconstruction],
                captions=["input", "reconstruction", "difference"],
                global_step=trainer.global_step,
            )
        if self.worst:
            log_images(
                logger=trainer.logger,
                key="worst_reconstruction",
                images=[worst_input, worst_reconstruction, worst_input - worst_reconstruction],
                captions=["input", "reconstruction", "difference"],
                global_step=trainer.global_step,
            )
