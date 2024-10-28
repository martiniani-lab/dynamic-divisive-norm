import torch
import pytorch_lightning as pl
import argparse
from model import rnn
from dataset import MnistDataModule
from config import get_config
import os
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    config, new_args = get_config()

    kwargs_dict = {}

    if new_args:
        for arg in new_args:
            if arg.startswith(("-", "--")):
                key = arg.lstrip("-")
                value = new_args[new_args.index(arg) + 1]
                # check the type of the value and change th type accordingly
                if value == "True":
                    value = True
                elif value == "False":
                    value = False
                elif value == "None":
                    value = None
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
                kwargs_dict[key] = value
    
    # model_name = "sMNIST_model_singular_val"
    model_name = config.MODEL_NAME
    start_from_checkpoint = config.CHECKPOINT
    logger = TensorBoardLogger(config.FOLDER_NAME, name=model_name)
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{config.FOLDER_NAME}/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )
    model = rnn(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        seq_length=config.SEQUENCE_LENGTH,
        learning_rate=config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES, 
        scheduler_change_step=config.SCHEDULER_CHANGE_STEP,
        scheduler_gamma=config.SCHEDULER_GAMMA,
        kwargs_dict=kwargs_dict,
    )
    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        permuted=config.PERMUTED,
        resize=config.RESIZE,
        seed=config.SEED
    )
    trainer = pl.Trainer(
        profiler=None,
        logger=logger,
        accelerator=config.ACCELERATOR,
        callbacks=[
                   LearningRateMonitor(logging_interval='epoch'),
                   ModelCheckpoint(save_top_k=-1, every_n_epochs=config.CHECKPOINT_EVERY_N_EPOCH),
                #    EarlyStopping(monitor="val_loss", check_finite=True),
        ],
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION
    )

    if start_from_checkpoint:
        version = config.VERSION
        checkpoint_folder = f"{config.FOLDER_NAME}/{model_name}/version_{version}/checkpoints/"
        checkpoint_files = os.listdir(checkpoint_folder)
        epoch_idx = [int(file.split('epoch=')[1].split('-')[0]) for file in checkpoint_files]
        max_idx = epoch_idx.index(max(epoch_idx))
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_files[max_idx])
        trainer.fit(model, dm, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)