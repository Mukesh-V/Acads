from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

import torch
from model import CNN
import pytorch_lightning as pl

data_transform = transforms.Compose([
  transforms.RandomCrop(size=256, pad_if_needed=True),
  transforms.ToTensor()
])
data = ImageFolder(root='/content/inaturalist_12K/train', transform=data_transform)
testdata = ImageFolder(root='/content/inaturalist_12K/val', transform=data_transform)

train_size = int(0.8 * len(data))
validation_size = len(data) - train_size
train_dataset, validation_dataset = random_split(data, [train_size, validation_size])

trainloader = DataLoader(train_dataset, 32, True)
valloader = DataLoader(validation_dataset, 32)
testloader = DataLoader(testdata, 32)

def train():
    wandb.init(project="FundDL-AS2")
    config = wandb.config
    wandb.run.name = "nf_{}-{}x_f_{}_{}_{}_fc_{}_dr_{}".format(config.nf, config.org, config.filter, config.activation, config.batch_norm, config.dense, config.drop)
    model = CNN(wandb.config)
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    wandb_logger = WandbLogger(log_model="all")
    trainer = pl.Trainer(max_epochs=config.epochs, precision=16, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    trainer.test(model=model, dataloaders=testloader)
    wandb.run.finish()

sweep_config = {
  "name": "Sweep-CNN",
  "method": "grid",
  "project": "FundDL-AS2",
  "metric":{
      "name":"val_accuracy",
      "goal":"maximize"
  },
  "parameters": {
        "epochs": {
            "values": [10]
        },
        "nf": {
            "values": [32, 64]
        },
        "org":{
            "values": [1, 2]
        },
        "filter":{
            "values": [2, 3]
        },
        "activation": {
            "values": ['relu']
        },
        "batch_norm": {
            "values": [True, False]
        },
        "drop":{
            "values": [0.2, 0.3]
        },
        "dense":{
            "values": [64]
        }
    }
}

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train, count=5)