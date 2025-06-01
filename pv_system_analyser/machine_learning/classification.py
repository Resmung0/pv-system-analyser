import lightning.pytorch as pl
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import F1Score

from pv_system_analyser.factory import ModelFactory

from .loss import FocalLoss

ModelFactory.register("cnn")


class CNNModule(pl.LightningModule):
    def __init__(
        self, n_classes: int, learning_rate: float, fl_alpha: float, fl_gamma: float
    ) -> None:
        super().__init__()

        # Layers
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2688, 960)
        self.fc2 = nn.Linear(960, n_classes)

        # Compilers
        self.learning_rate = learning_rate
        self.loss_fn = FocalLoss(fl_alpha, fl_gamma)
        self.metric = F1Score(task="multiclass", num_classes=n_classes, average="macro")

    def forward(self, x):
        # Initial dimentions --> N, 1, 40, 24
        x = nn.functional.relu(self.conv1(x))  # N, 64, 38, 22
        x = nn.functional.relu(self.conv2(x))  # N, 64, 36, 20
        x = self.pool(x)  # N, 64, 18, 10

        x = nn.functional.relu(self.conv3(x))  # N, 128, 16, 8
        x = nn.functional.relu(self.conv4(x))  # N, 128, 14, 6
        x = self.pool(x)  # N, 128, 7, 3

        x = nn.functional.dropout(self.flatten(x))  # N, 2688
        x = nn.functional.relu(self.fc1(x))  # N, 960
        return self.fc2(x)  # N, 12

    def training_step(self, batch, _):
        loss, metric = self.__common_step(batch)
        self.__log_metrics_and_loss(metric, loss, "train")
        return loss

    def validation_step(self, batch, _):
        loss, metric = self.__common_step(batch)
        self.__log_metrics_and_loss(metric, loss, "val")
        return loss

    def test_step(self, batch, _):
        loss, metric = self.__common_step(batch)
        self.__log_metrics_and_loss(metric, loss, "test")
        return loss

    def predict_step(self, batch, _):
        scores = self.forward(batch)
        scores = scores.argmax(dim=1)
        self.log("predict", scores)
        return scores

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1_score",
                "frequency": "trainer.check_val_every_n_epoch",
            },
        }

    def __common_step(self, batch) -> tuple[Tensor, Tensor]:
        image, label = batch
        scores = self.forward(image)
        loss = self.loss_fn(scores, label)
        metric_score = self.metric(scores.argmax(dim=1), label.argmax(dim=1))
        return loss, metric_score

    def __log_metrics_and_loss(self, metric: Tensor, loss: Tensor, stage: str) -> None:
        params = dict(on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_loss", loss, **params)
        self.log(f"{stage}_f1_score", metric, **params)


class AutoencoderModule(pl.LightningModule):
    def __init__(self, latent_dim, n_classes, learning_rate):
        super().__init__(n_classes, learning_rate)

        # Encoder Layers
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2)

        # Latent layer
        self.latent_dim = latent_dim

        # Decoder Layer
        self.upconv1 = nn.ConvTranspose2d(128, 128, 3)
        self.deconv1 = nn.Conv2d(128, 64, 3)
        self.upconv2 = nn.ConvTranspose2d(64, 64, 3)
        self.deconv2 = nn.Conv2d(64, 1, 3)

        # Compilers
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.f1_score = F1Score(
            task="multiclass", num_classes=n_classes, average="macro"
        )

    def encoder(self, x):
        # Initial dimentions --> N, 1, 40, 24
        x = nn.functional.relu(self.conv1(x))  # N, 64, 38, 22
        x = nn.functional.relu(self.conv2(x))  # N, 64, 36, 20
        x = self.pool(x)  # N, 64, 18, 10
        x = nn.functional.relu(self.conv3(x))  # N, 128, 16, 8
        x = nn.functional.relu(self.conv4(x))  # N, 128, 14, 6
        x = self.pool(x)  # N, 128, 7, 3
        return x

    def decoder(self, x):
        # Up-sample and apply convolutions
        x = nn.functional.relu(self.upconv1(x))
        x = nn.functional.relu(self.deconv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.upconv2(x))
        # Use sigmoid for image reconstruction (0-1 range)
        x = nn.functional.sigmoid(self.deconv2(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        image, label = batch
        scores = self.forward(image)
        loss = self.loss_fn(scores, label)
        self.log("train_loss", loss)
        return loss


def validation_step(self, batch, batch_idx):
    image, label = batch
    scores = self.forward(image)
    loss = self.loss_fn(scores, label)
    self.log("val_loss", loss)

    preds = torch.argmax(scores, dim=1)
    f1_score = self.f1_score(preds, label)
    self.log("val_f1_score", f1_score)

    return loss


def test_step(self, batch, batch_idx):
    image, label = batch
    scores = self.forward(image)
    loss = self.loss_fn(scores, label)
    self.log("test_loss", loss)

    preds = torch.argmax(scores, dim=1)
    f1_score = self.f1_score(preds, label)
    self.log("test_f1_score", f1_score)

    return loss


def predict_step(self, batch, batch_idx, dataloader_idx=None):
    image = batch
    scores = self.forward(image)
    return torch.argmax(scores, dim=1)


def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.learning_rate)
