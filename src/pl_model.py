import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics


class XraySegmentationModel(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.backbone_name = hparams.backbone_name
        self.model = smp.Unet(
            encoder_name=self.backbone_name,
            **hparams["model_args"],
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.iou_covid = torchmetrics.JaccardIndex(num_classes=2, ignore_index=0)
        self.iou_lung = torchmetrics.JaccardIndex(num_classes=2, ignore_index=0)

    def forward(self, image):
        model_output = self.model(image)
        return model_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        image, covid_mask, lung_mask = batch["image"], batch["covid_mask"], batch["lung_mask"]

        pred_mask = self(image)

        loss = self.criterion(pred_mask, torch.stack((covid_mask, lung_mask), dim=1))

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        image, covid_mask, lung_mask = batch["image"], batch["covid_mask"], batch["lung_mask"]

        pred_mask = self(image)
        stacked_target_masks = torch.stack((covid_mask, lung_mask), dim=1)
        loss = self.criterion(pred_mask, stacked_target_masks)

        self.iou_covid(torch.sigmoid(pred_mask[:, 0, :]), covid_mask.long())
        self.iou_lung(torch.sigmoid(pred_mask[:, 1, :]), lung_mask.long())

        self.log("valid_loss", loss, on_step=True, on_epoch=False)
        self.log("iou_covid", self.iou_covid, on_step=False, on_epoch=True, prog_bar=True)
        self.log("iou_lung", self.iou_lung, on_step=False, on_epoch=True, prog_bar=True)
