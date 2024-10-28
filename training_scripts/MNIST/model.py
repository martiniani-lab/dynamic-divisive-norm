import torch
import torch.nn.functional as F
import models.fixed_point as organics
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision
from collections import defaultdict
import torch.nn.utils.parametrize as P


class feedforward(pl.LightningModule):
    def __init__(self, input_size, ff_input_size, hidden_sizes, learning_rate, num_classes, scheduler_change_step, scheduler_gamma, kwargs_dict):
        super().__init__()
        self.input_size = input_size
        self.ff_input_size = ff_input_size
        self.hidden_sizes = hidden_sizes

        self.lr = learning_rate
        self.scs = scheduler_change_step
        self.gamma = scheduler_gamma

        self.autoencoder = organics.Autoencoder(input_dim=input_size, out_dim=ff_input_size)
        self.org1 = organics.ff(input_size=ff_input_size, output_size=hidden_sizes[0], **kwargs_dict)
        self.org2 = organics.ff(input_size=hidden_sizes[0], output_size=hidden_sizes[1], **kwargs_dict)

        self.fc = nn.Linear(hidden_sizes[1], num_classes)

        self.autoencoder_loss_fn = nn.MSELoss()
        self.classification_loss_fn = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        self.training_step_outputs = []

        self.activations = defaultdict(list)

        self.save_hyperparameters()

    def forward(self, x):
        x, y = self.autoencoder(x)
        x = self.org1(x)
        x = self.org2(x)
        return self.fc(x), y

    def training_step(self, batch, batch_idx):
        x, target = batch
        loss, scores, target = self._common_step(batch, batch_idx)

        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        if batch_idx == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("mnist_images", grid, self.global_step)

        self.training_step_outputs.append(
            {"loss": loss, "scores": scores, "target": target}
        )

        return {"loss": loss, "scores": scores, "target": target}

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        scores = torch.cat([x["scores"] for x in outputs])
        target = torch.cat([x["target"] for x in outputs])
        self.log_dict(
            {
                "train_acc": self.accuracy(scores, target),
                "train_f1": self.f1_score(scores, target),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_acc": self.accuracy(scores, target),
                "val_f1": self.f1_score(scores, target),
                "val_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "test_acc": self.accuracy(scores, target),
                "test_f1": self.f1_score(scores, target),
                "test_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # save the time series for the first batch
        if batch_idx == 0:
            x, _ = batch
            # x = x.reshape(x.size(0), self.seq_length, -1)
            x = self._transform_inputs(x)
            # the forward method
            x, y = self.autoencoder(x)
            self.activations["org1_input"] = x
            x = self.org1(x)
            self.activations["org1_output"] = x
            self.activations["org2_input"] = x
            x = self.org2(x)
            self.activations["org2_output"] = x
        return loss

    def _common_step(self, batch, batch_idx):
        x, target = batch
        # convert into sequential task
        x = self._transform_inputs(x)
        scores, y = self.forward(x)
        loss = self.classification_loss_fn(scores, target) + self.autoencoder_loss_fn(x, y)
        return loss, scores, target        

    def predict_step(self, batch, batch_idx):
        x, target = batch
        x = self._transform_inputs(x)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def _transform_inputs(self, x):
        x = x.reshape(x.size(0), self.input_size)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.scs, gamma=self.gamma)  # Define scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
            }
        }

