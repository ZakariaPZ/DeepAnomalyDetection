import typing as th
import lightning_toolbox
import lightning.pytorch as pl
import torchmetrics
import dypy as dy
import torch


class NoveltyAUROCCallback(pl.Callback):
    def __init__(
        self,
        objective: th.Union["lightning_toolbox.Objective", str, None] = "objective",
        objective_recompute: bool = False,
        objective_key: str = "loss",
        objective_cls: th.Optional[th.Type[lightning_toolbox.Objective]] = "lightning_toolbox.Objective",
        objective_args: th.Optional[th.Dict] = None,
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

        # metrics
        self.val_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1)
        self.test_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1)

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

    def on_validation_batch_end(self, **kwargs) -> None:
        """Called when the validation batch ends."""
        self.step(**kwargs, name="val")

    def on_test_batch_end(self, **kwargs) -> None:
        """Called when the test batch ends."""
        self.step(**kwargs, name="test")

    def step(self, batch, pl_module: "pl.LightningModule", name: str = "val", **kwargs):
        is_val = name == "val"
        if self.__objective_recompute:
            objective_results = self.objective(batch=batch, **kwargs, training_module=pl_module, return_factors=False)
        else:
            objective_results = self.objective.results

        scores = objective_results[self.__objective_key]
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
