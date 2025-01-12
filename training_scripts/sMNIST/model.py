import torch
import torch.nn.functional as F
import models.dynamical as organics
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision
from collections import defaultdict
import torch.nn.utils.parametrize as P


class rnn(pl.LightningModule):
    def __init__(self, input_size, hidden_size, seq_length, learning_rate, num_classes, scheduler_change_step, scheduler_gamma, kwargs_dict):
        super().__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.lr = learning_rate
        self.scs = scheduler_change_step
        self.gamma = scheduler_gamma
        # self.org = organics.rnnCell(input_size=input_size, hidden_size=hidden_size, **kwargs_dict)
        self.org = organics.rnnCell_unrectified(input_size=input_size, hidden_size=hidden_size, **kwargs_dict)

        self.fc = nn.Linear(hidden_size, num_classes)
        # Initial hidden states for y and a neurons
        self.register_buffer("y0", torch.rand((hidden_size)))
        self.register_buffer("a0", torch.rand((hidden_size)))
        # self.register_buffer("b00", torch.rand((hidden_size)))
        self.register_buffer("b00", torch.ones((hidden_size)))
        self.register_buffer("b10", torch.rand((hidden_size)))
        # self.register_buffer("x_max", torch.sqrt(torch.tensor([self.input_size])))

        self.y0 = self.y0 / torch.norm(self.y0)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        
        self.activations = defaultdict(list)

        self.save_hyperparameters()

    def forward(self, x, y, a, b0, b1):
        # Add code to set max singular value of Wr to a max value 1
        # with P.cached():
        #     for i in range(self.seq_length):
        #         y, a, b0, b1 = self.org(x[:, i, :], y, a, b0, b1)
        for i in range(self.seq_length):
            y, a, b0, b1 = self.org(x[:, i, :], y, a, b0, b1)
        # y = self.org.get_activation_y(y)
        # a = self.org.get_activation_a(a)
        # b = self.org.get_activation_a(b)
        return self.fc(y)

    def training_step(self, batch, batch_idx):
        x, target = batch
        loss, scores, target = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, target)

        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        # save train loss and accuracy
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        self.logger.experiment.add_scalar("train_accuracy", accuracy, self.global_step)

        if batch_idx == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, int((self.seq_length * self.input_size) ** 0.5), int((self.seq_length * self.input_size) ** 0.5)))
            self.logger.experiment.add_image("mnist_images", grid, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_acc": self.accuracy(scores, target),
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
            y = self.y0.repeat(x.size(0), 1)
            a = self.a0.repeat(x.size(0), 1)
            b0 = self.b00.repeat(x.size(0), 1)
            b1 = self.b10.repeat(x.size(0), 1)
            self.activations["y"].append(y)
            self.activations["a"].append(a)
            self.activations["b0"].append(b0)
            self.activations["b1"].append(b1)
            for i in range(self.seq_length):
                y, a, b0, b1 = self.org(x[:, i, :], y, a, b0, b1)
                self.activations["x"].append(x[:, i, :])
                self.activations["y"].append(y)
                self.activations["a"].append(a)
                self.activations["b0"].append(b0)
                self.activations["b1"].append(b1)
            self.activations["target"].append(target)
            self.activations["preds"].append(torch.argmax(scores, dim=1))
        return loss

    def _common_step(self, batch, batch_idx):
        x, target = batch
        # convert into sequential task
        x = self._transform_inputs(x)
        scores = self.forward(x, self.y0.repeat(x.size(0), 1), self.a0.repeat(x.size(0), 1), self.b00.repeat(x.size(0), 1), self.b10.repeat(x.size(0), 1))
        loss = self.loss_fn(scores, target)
        return loss, scores, target

    def predict_step(self, batch, batch_idx):
        x, target = batch
        x = self._transform_inputs(x)
        scores = self.forward(x, self.y0.repeat(x.size(0), 1), self.a0.repeat(x.size(0), 1), self.b00.repeat(x.size(0), 1), self.b10.repeat(x.size(0), 1))
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def _transform_inputs(self, x):
        x = x.reshape(x.size(0), self.seq_length, self.input_size)
        # x = x / self.x_max
        return x
    
    # def on_after_backward(self):
    #     """With gradient clipping"""
    #     total_norm_before = 0
    #     max_norm = 0.25

    #     # Calculate the total gradient norm before clipping
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             param_norm = param.grad.data.norm(2)
    #             total_norm_before += param_norm.item() ** 2

    #     total_norm_before = total_norm_before ** 0.5

    #     # Clip the gradients across all parameters
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    #     # Calculate the total gradient norm after clipping
    #     total_norm_after = 0
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             param_norm = param.grad.data.norm(2)
    #             total_norm_after += param_norm.item() ** 2

    #     total_norm_after = total_norm_after ** 0.5

    #     # Log both norms
    #     self.logger.experiment.add_scalar("grad_total_norm_before_clipping", total_norm_before, self.global_step)
    #     self.logger.experiment.add_scalar("grad_total_norm_after_clipping", total_norm_after, self.global_step)

    def on_after_backward(self):
        """Without gradient clipping."""
        total_norm = 0

        # Calculate the total gradient norm before clipping
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5

        self.logger.experiment.add_scalar("gradient_norm", total_norm, self.global_step)

        # Log the spectral radius of Wr
        eigvals = torch.linalg.eigvals(self.org.Wr())
        spectral_radius = torch.max(torch.abs(eigvals)).item()
        self.logger.experiment.add_scalar("spectral_radius_Wr", spectral_radius, self.global_step)
        self.log("spectral_radius_Wr", spectral_radius, on_step=True, on_epoch=False, prog_bar=True)
        
        # Log the spectral radius of Wzx
        spectral_norm = torch.svd(self.org.Wzx()).S[0].item()
        self.logger.experiment.add_scalar("spectral_norm_Wzx", spectral_norm, self.global_step)
        self.log("spectral_norm_Wzx", spectral_norm, on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.scs, gamma=self.gamma)  # Define scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
            }
        }


    
