import json
from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image
from torchvision.datasets import VisionDataset
from torchvision import datasets

class WSSSDataset(VisionDataset):
    """Custom Dataset for WSSS Histopathology Images."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader=datasets.folder.default_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self._split = split
        self._base_folder = Path(self.root) / "wsss"
        self._classes = [
            "Normal",
            "Stroma",
            "Tumor"
        ]
        
        self._class_to_idx = {cls_name: i for i, cls_name in enumerate(self._classes)}
        
        # Define data paths
        self._data_folder = self._base_folder / split

        self._labels = []
        self._image_files = []
        
        for class_label in self._classes:
            class_folder = self._data_folder / class_label
            if not class_folder.exists():
                raise RuntimeError(f"Class folder {class_folder} not found.")
            image_files = list(class_folder.glob("*.png"))
            self._labels += [self._class_to_idx[class_label]] * len(image_files)
            self._image_files += image_files

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image, label = self.loader(self._image_files[idx]), self._labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    @property
    def classes(self):
        return self._classes

    def extra_repr(self) -> str:
        return f"split={self._split}, root={self.root}"
