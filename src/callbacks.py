import io
import itertools
import typing as th

import dypy as dy
import lightning.pytorch as pl
import lightning_toolbox
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchmetrics
from torchvision import transforms
from torchvision.utils import make_grid


class NoveltyAUROCCallback(pl.Callback):
    def __init__(
        self,
        objective: th.Union["lightning_toolbox.Objective", str, None] = "objective",
        objective_recompute: bool = False,
        objective_key: str = "loss",
        objective_cls: th.Optional[th.Type[lightning_toolbox.Objective]] = "lightning_toolbox.Objective",
        objective_args: th.Optional[th.Dict] = None,
        score_negative: bool = False,  # if True, the score is the negative of the objective
    ):
        super().__init__()
        assert (
            not objective_recompute or objective is not None
        ), "objective_recompute requires a objective bound to the training_module"
        self.__objective_descriptor = objective
        self.__objective_cls = objective_cls
        self.__objective_args: th.Optional[dict] = objective_args
        self.__objective_key: str = objective_key
        self.__objective_recompute: bool = objective_recompute
        self.objective: lightning_toolbox.Objective = None  # type: ignore # will be set in setup
        self.score_negative = score_negative

        # metrics
        self.val_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1, task="binary")
        self.test_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1, task="binary")

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if stage not in ["fit", "test"]:
            return
        if self.__objective_descriptor is not None:
            self.objective: lightning_toolbox.Objective = (
                dy.eval(self.__objective_descriptor, pl_module)
                if isinstance(self.__objective_descriptor, str)
                else self.__objective_descriptor
            )
        else:
            self.objective = dy.get_value(self.__objective_cls)(**self.__objective_args)

        # bind metrics to pl_module
        pl_module.val_auroc = self.val_auroc
        pl_module.test_auroc = self.test_auroc

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: th.Optional["STEP_OUTPUT"],
        batch: th.Any,
        batch_idx: int,
        dataloader_idx: th.Optional[int] = None,
    ) -> None:
        """Called when the validation batch ends."""
        self.step(
            batch=batch,
            pl_module=pl_module,
            outputs=outputs,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            trainer=trainer,
            name="val",
        )

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: th.Optional["STEP_OUTPUT"],
        batch: th.Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""
        self.step(
            batch=batch,
            pl_module=pl_module,
            outputs=outputs,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            trainer=trainer,
            name="test",
        )

    def step(self, batch, pl_module: "pl.LightningModule", name: str = "val", **kwargs):
        is_val = name == "val"
        if self.__objective_recompute:
            objective_results = self.objective(batch=batch, **kwargs, training_module=pl_module, return_factors=False)
        else:
            objective_results = self.objective.results

        scores = objective_results[self.__objective_key] * (-1 if self.score_negative else 1)
        labels: torch.Tensor = batch[1]
        metric = self.val_auroc if is_val else self.test_auroc
        metric(
            preds=scores.reshape(-1),  # auroc expects predictions to have higher values for the positive class
            target=labels.reshape(-1),
        )
        normal_scores = scores[labels == 1]
        anomaly_scores = scores[labels == 0]
        results = objective_results if self.__objective_recompute else {}
        if normal_scores.shape[0] != 0:
            results["novelty_auroc/score/normal"] = normal_scores.mean()
        if anomaly_scores.shape[0] != 0:
            results["novelty_auroc/score/anomaly"] = anomaly_scores.mean()
        if anomaly_scores.shape[0] != 0 and normal_scores.shape[0] != 0:
            results["novelty_auroc/score/difference"] = normal_scores.mean() - anomaly_scores.mean()
        results["novelty_auroc"] = metric
        self.log_results(pl_module, results, name=name)

        if is_val:
            # get the points on the ROC curve
            fpr, tpr, thresholds = torchmetrics.functional.roc(
                preds=scores.reshape(-1),
                target=labels.reshape(-1),
                task="binary",
            )
            # find the product of tpr, (1-fpr)
            product = tpr * (1 - fpr)
            # get the threshold with the highest product
            i = torch.argmax(product)
            # save the threshold for later
            pl_module.threshold = thresholds[i]

        return results

    def log_results(self, pl_module: pl.LightningModule, results: th.Dict, name: str = "val"):
        is_val = name == "val"
        for item, value in results.items():
            pl_module.log(
                f"{item}/{name}",
                value.mean() if isinstance(value, torch.Tensor) else value,
                on_step=not is_val,
                on_epoch=is_val,
                logger=True,
                sync_dist=True,
                prog_bar=is_val and name == "loss",
            )


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
        outputs: th.Optional["STEP_OUTPUT"],
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
            pl_module.logger.experiment.add_image(
                "best_reconstruction",
                make_grid(
                    [best_input, best_reconstruction, best_input - best_reconstruction],
                ),
                global_step=trainer.global_step,
            )
        if self.worst:
            pl_module.logger.experiment.add_image(
                "worst_reconstruction",
                make_grid(
                    [worst_input, worst_reconstruction, worst_input - worst_reconstruction],
                ),
                global_step=trainer.global_step,
            )


class ConfusionMatrixCallback(pl.Callback):
    def __init__(
        self,
        num_classes: int = 10,
    ):
        super().__init__()

        self.num_classes = num_classes

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.cm = torch.zeros(self.num_classes, self.num_classes).cpu()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: th.Optional["STEP_OUTPUT"],
        batch: th.Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, y = batch
        reconstruction_loss = torch.nn.functional.mse_loss(x, pl_module(x), reduction="none").mean(dim=(1, 2, 3))
        # get the predicted class
        y_hat = torch.where(
            torch.sigmoid(-1 * reconstruction_loss) < pl_module.threshold,
            torch.zeros_like(y),
            torch.ones_like(y),
        )
        # get the confusion matrix

        cm = torchmetrics.functional.confusion_matrix(y_hat, y, num_classes=self.num_classes, task="multiclass")
        # add the confusion matrix to the total confusion matrix
        pl_module.cm += cm.cpu()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        def plot_confusion_matrix(cm, class_names):
            figure = plt.figure(figsize=(8, 8))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion matrix")
            plt.colorbar()
            tick_marks = range(len(class_names))
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)

            # Use white text if squares are dark; otherwise black.
            threshold = cm.max() / 2.0

            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

            plt.tight_layout()
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            return figure

        def plot_to_image(figure):
            buf = io.BytesIO()

            # Use plt.savefig to save the plot to a PNG in memory.
            plt.savefig(buf, format="jpeg")

            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(figure)
            buf.seek(0)

            image = transforms.ToTensor()(PIL.Image.open(buf))

            return image

        # log the confusion matrix
        pl_module.logger.experiment.add_image(
            "confusion_matrix",
            plot_to_image(
                plot_confusion_matrix(
                    pl_module.cm,
                    class_names=range(self.num_classes),
                ),
            ),
            global_step=trainer.global_step,
        )
