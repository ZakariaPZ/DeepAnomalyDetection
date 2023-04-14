import typing as th
from lightning_toolbox.data import DataModule
from lightning_toolbox.data.module import transform_dataset
import torch
import numpy as np


class NoveltyBaseDataset(torch.utils.data.Dataset):
    def __init__(self, normal_targets: th.Optional[None], relabel: bool = False):
        self.normal_targets = np.array(normal_targets if normal_targets is not None else [])
        self.relabel = relabel

    def __len__(self):
        return len(self.dataset) if not hasattr(self, "indices") else len(self.indices)

    def __getitem__(self, index):
        if hasattr(self, "indices"):
            index = self.indices[index]
        if self.relabel:
            inputs, target = self.dataset[index]
            return inputs, 1 if target in self.normal_targets else 0
        return self.dataset[index]


class KeepNormalDataset(NoveltyBaseDataset):
    """Filters out the normal samples from the dataset."""

    def __init__(self, original_dataset: torch.utils.data.Dataset, normal_targets: th.Optional[th.List] = None):
        super().__init__(normal_targets, relabel=False)
        self.dataset = (
            original_dataset if not isinstance(original_dataset, torch.utils.data.Subset) else original_dataset.dataset
        )
        self.indices = np.array(
            np.arange(len(original_dataset))
            if not isinstance(original_dataset, torch.utils.data.Subset)
            else original_dataset.indices
        )
        if len(normal_targets):
            indices_map = np.zeros(len(self.dataset), dtype=int)
            indices_map[self.indices] = 1
            self.indices = np.where(indices_map & np.isin(self.dataset.targets, self.normal_targets))[0]


class IsNormalDataset(NoveltyBaseDataset):
    """Labels the samples as normal (1) or not normal. (0)"""

    def __init__(self, original_dataset: torch.utils.data.Dataset, normal_targets: th.Optional[th.List] = None):
        super().__init__(normal_targets, relabel=True)
        self.dataset = (
            original_dataset if not isinstance(original_dataset, torch.utils.data.Subset) else original_dataset.dataset
        )
        self.indices = np.array(
            np.arange(len(self.dataset))
            if not isinstance(original_dataset, torch.utils.data.Subset)
            else original_dataset.indices
        )


class NoveltyDetectionDatamodule(DataModule):
    def __init__(
        self,
        # normal/anomaly
        normal_targets: th.Optional[th.List[int]] = None,
        **kwargs  # see lightning_toolbox.data.DataModule for more details
    ):
        super().__init__(**kwargs)
        self.normal_targets = normal_targets

    def setup(self, stage: str = None):
        super().setup(
            stage, transform=False
        )  # this will create self.train_dataset, self.val_dataset, self.test_dataset

        if stage == "fit" and self.train_dataset is not None and self.normal_targets is not None:
            self.train_data = KeepNormalDataset(self.train_data, self.normal_targets)
            self.train_data = transform_dataset(self.train_data, self.train_transforms)
        if stage == "fit" and self.val_dataset is not None and self.normal_targets is not None:
            self.val_data = IsNormalDataset(self.val_data, self.normal_targets)
            self.val_data = transform_dataset(self.val_data, self.val_transforms)
        if stage == "test" and self.test_dataset is not None and self.normal_targets is not None:
            self.test_data = IsNormalDataset(self.test_data, self.normal_targets)
            self.test_data = transform_dataset(self.test_data, self.test_transforms)
