from typing import Literal

import lightning.pytorch as pl
from dvclive.lightning import DVCLiveLogger
from torchvision.transforms import (
    Compose, RandomChoice,
    RandomAdjustSharpness, RandomEqualize,
    RandomHorizontalFlip, RandomVerticalFlip   
)

from pv_system_analyser.factory import ModelFactory, DataModuleFactory


def train_model(
    problem: Literal['classification', 'segmentation'],
    model_name: str,
    augmentations: Compose,
    batch_size: int,
    num_workers: int,
    epochs: int,
    **kwargs
) -> pl.Trainer:
    
    # Get model
    model = ModelFactory.create(model_name, mode='training', **kwargs)

    # Get data
    data_module = DataModuleFactory.create(
        problem,
        augmentations=augmentations,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Train the model
    trainer = _setup_trainer(epochs)
    trainer.fit(model, data_module)
    return trainer

def _setup_trainer(epochs: int) -> pl.Trainer:
    logger = DVCLiveLogger(
        save_dvc_exp=True, log_model=True, dir="logs/classifier"
    )
    
    callbacks = [
        pl.callbacks.EarlyStopping("val_f1_score", patience=5, mode="max"),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            every_n_epochs=10,
            dirpath="assets/models/classifier",
            monitor="val_f1_score",
            mode="max",
        ),
    ]
    
    trainer = pl.Trainer(
        profiler="simple",
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        callbacks=callbacks,
        logger=logger,
    )
    return trainer 





if "__main__" == __name__:
    config = {
        'experiment': {
            'problem': 'classification',
            'model_name': 'cnn',
            "augmentations": Compose(
                RandomChoice((RandomHorizontalFlip, RandomVerticalFlip), p=0.5),
                RandomChoice((RandomAdjustSharpness, RandomEqualize), p=0.5)
            ),
            "batch_size": 32,
            "num_workers": 1,
            "epochs": 100
        },
        'model': {
            'n_classes': 10,
            "learning_rate": 0.001,
            "fl_alpha": 0.25, 
            "fl_gamma": 2.0
        },

    }
    
    trainer = train_model(**config["experiment"], **config["model"])