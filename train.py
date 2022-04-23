import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data_utils import XRayDataset, generate_transforms, get_config
from src.pl_model import XraySegmentationModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, default="unet_mobilenet_v2.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None, choices=[None, "ddp", "ddp_cpu", "dp"])
    parser.add_argument("--fast_dev_run", dest="fast_dev_run", action="store_true")
    parser.add_argument("--resume_from_checkpoint", dest="resume_from_checkpoint", action="store_true")
    parser.set_defaults(fast_dev_run=False, use_comet=False, resume_from_checkpoint=False)

    args = parser.parse_args()

    assert args.gpu_id is None or args.gpus is None

    pl.seed_everything(args.seed)

    print("Use {} config".format(args.experiment_config))
    config = get_config(args.experiment_config)

    train_data_dir = Path(config.root_data_path + "Train")
    test_data_dir = Path(config.root_data_path + "Test")
    valid_data_dir = Path(config.root_data_path + "Val")

    train_transforms = generate_transforms("train")
    valid_transforms = generate_transforms("valid")
    test_transforms = generate_transforms("test")

    train_dataset = XRayDataset(train_data_dir, config.backbone_name, **config.datasets, transforms=train_transforms)
    valid_dataset = XRayDataset(valid_data_dir, config.backbone_name, **config.datasets, transforms=valid_transforms)
    test_dataset = XRayDataset(test_data_dir, config.backbone_name, **config.datasets, transforms=test_transforms)

    loaders = {
        "train_dataloaders": torch.utils.data.DataLoader(train_dataset, shuffle=True, **config.loaders),
        "val_dataloaders": torch.utils.data.DataLoader(valid_dataset, shuffle=False, **config.loaders),
    }
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **config.loaders)

    pl_model = XraySegmentationModel(config)

    logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    if "logger" in config and args.use_comet:
        comet_logger = pl.loggers.CometLogger(**config["logger"], experiment_name=config.experiment_name)
        logger = [logger, comet_logger]

    gpus = None
    if args.gpu_id is not None:
        gpus = [
            args.gpu_id,
        ]
    if args.gpus is not None:
        gpus = args.gpus

    dirname = f"{config.ckeckpoints_dir}/{config.experiment_name}_seed_{str(args.seed)}/"
    filename = config.experiment_name + "_seed_" + str(args.seed) + "_{epoch}-{iou_total:.3f}-{valid_loss:.3f}"
    checkpoint_callback = ModelCheckpoint(
        filename=filename, dirpath=dirname, save_top_k=1, monitor="valid_loss", verbose=True, mode="max"
    )

    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        resume_from_checkpoint = dirname + "/" + os.listdir(dirname)[0]

    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        gpus=gpus,
        strategy=args.strategy,
        fast_dev_run=args.fast_dev_run,
        resume_from_checkpoint=resume_from_checkpoint,
        callbacks=[
            checkpoint_callback,
        ],
    )
    trainer.fit(pl_model, **loaders)
    checkpoint_path = dirname + "/" + os.listdir(dirname)[0]
    model = XraySegmentationModel.load_from_checkpoint(checkpoint_path)
    trainer.test(model, test_loader)
