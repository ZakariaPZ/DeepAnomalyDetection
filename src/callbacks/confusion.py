import io
import itertools
import typing as th
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import PIL
import torch
import torchmetrics
from torchvision import transforms
from .utils import log_images


class ConfusionMatrixCallback(pl.Callback):
    def __init__(
        self, num_classes: int = 10, threshold_lookup: str = "auroc", score: str = "loss", negative_score: bool = False
    ):
        super().__init__()
        self.threshold_lookup = threshold_lookup
        self.num_classes = num_classes
        self.score = score
        self.negative_score = negative_score

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.cm = torch.zeros(self.num_classes, self.num_classes)

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
        scores = pl_module.objective.results[self.score]

        # get the predicted class
        y_hat = torch.where(
            torch.sigmoid(-1 * scores) < pl_module.threshold,
            torch.zeros_like(y),
            torch.ones_like(y),
        )
        # get the confusion matrix

        cm = torchmetrics.functional.confusion_matrix(y_hat, y, num_classes=self.num_classes, task="multiclass")
        # add the confusion matrix to the total confusion matrix
        pl_module.cm += cm

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

        imgs = plot_to_image(
            plot_confusion_matrix(
                pl_module.cm,
                class_names=range(self.num_classes),
            ),
        )
        log_images(
            logger=trainer.logger,
            key="confusion_matrix",
            imgs=[imgs],
            global_step=trainer.global_step,
        )
