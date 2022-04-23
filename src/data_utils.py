import os
from dataclasses import dataclass
from json import load
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import omegaconf
import torch
from PIL import Image
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import Dataset


@dataclass
class DataSample:
    sample_label: str
    image_path: Path
    infection_mask_path: Path
    lung_mask_path: Path


def get_class_by_name(module, model_name: str):
    model = None
    if hasattr(module, model_name):
        model = getattr(module, model_name)
    return model


def get_config(exp_path: str, cfg_path: str = "./configs/"):
    exp_path_dir = cfg_path + "experiments/"
    files = [cfg_path + fl for fl in os.listdir(cfg_path) if "yaml" in fl]

    base_cfg = [omegaconf.OmegaConf.load(cfg) for cfg in files]
    base_cfg = omegaconf.OmegaConf.merge(*base_cfg)

    exp_cfg = omegaconf.OmegaConf.load(exp_path_dir + exp_path)
    cfg = omegaconf.OmegaConf.merge(base_cfg, exp_cfg)
    return cfg


def load_image(img_path: str, convert_to_rgb: bool = False) -> np.ndarray:
    im = Image.open(img_path)
    if convert_to_rgb:
        im = im.convert("RGB")
    im = np.array(im, dtype=np.uint8)
    return im


def generate_transforms(mode: str = "train"):
    if mode == "train":
        return A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
                    ],
                    p=0.8,
                ),
                A.CLAHE(p=0.8),
            ],
            additional_targets={"mask1": "mask", "mask2": "mask"},
        )
    elif mode in ["valid", "test"]:
        return None
    else:
        raise ValueError("Unsupported mode type")


class XRayDataset(Dataset):
    def __init__(
        self,
        root_data_dir: str,
        model_name: str,
        pretrained: str = "imagenet",
        dataset_usage: str = "COVID-19",
        use_sample_label: bool = False,
        transforms: Optional[A.Compose] = None,
    ) -> None:
        self.transforms = transforms
        self.use_sample_label = use_sample_label
        self.dataset_usage = dataset_usage.split(",")
        self.dataset_usage_mapping = {t: i for i, t in enumerate(self.dataset_usage)}
        self.root_data_dir = Path(root_data_dir)

        self.sample_paths = []
        for dataset in self.dataset_usage:
            self.sample_paths.extend(self._get_all_images_paths(Path(self.root_data_dir, dataset), dataset))

        self.preprocess_input = get_preprocessing_fn(model_name, pretrained=pretrained)

    def _get_all_images_paths(self, root_data_dir: str, sample_label: str):
        images_dir = Path(root_data_dir, "images")
        infection_masks_dir = Path(root_data_dir, "infection masks")
        lung_masks_dir = Path(root_data_dir, "lung masks")

        images_path = list(images_dir.glob("*.png"))
        infection_masks_path = list(infection_masks_dir.glob("*.png"))
        lung_masks_path = list(lung_masks_dir.glob("*.png"))

        total_paths = [DataSample(sample_label, *t) for t in zip(images_path, infection_masks_path, lung_masks_path)]
        return total_paths

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index: int):
        sample = self.sample_paths[index]

        image = load_image(sample.image_path, convert_to_rgb=True)
        covid_mask = load_image(sample.infection_mask_path)
        lung_mask = load_image(sample.lung_mask_path)

        covid_mask = np.where(covid_mask > 0, 1, 0)
        lung_mask = np.where(lung_mask > 0, 1, 0)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask1=covid_mask, mask2=lung_mask)
            image = augmented["image"]
            covid_mask = augmented["mask1"]
            lung_mask = augmented["mask2"]

        image = self.preprocess_input(image)

        if not isinstance(image, torch.Tensor):
            image = torch.FloatTensor(image)
        if not isinstance(covid_mask, torch.Tensor):
            covid_mask = torch.from_numpy(covid_mask)
        if not isinstance(lung_mask, torch.Tensor):
            lung_mask = torch.from_numpy(lung_mask)

        image = image.to(torch.float32)
        covid_mask = covid_mask.to(torch.float32)
        lung_mask = lung_mask.to(torch.float32)
        image = image.permute(2, 0, 1)

        output = {"image": image, "covid_mask": covid_mask, "lung_mask": lung_mask}
        if self.use_sample_label:
            label = sample.sample_label
            label_id = self.dataset_usage_mapping[label]
            output["sample_label_id"] = torch.LongTensor(
                [
                    label_id,
                ]
            )
        return output
