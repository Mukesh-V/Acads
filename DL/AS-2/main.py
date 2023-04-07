from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from model_scratch import ScratchCNN
from model_resnet_mod import ModdedResnet50
import pytorch_lightning as pl

sweep_config = {
  "name": "Sweep-CNN",
  "method": "grid",
  "project": "FundDL-AS2",
  "metric":{
      "name":"val_accuracy",
      "goal":"maximize"
  },
  "parameters": {
        "model": {
            "values": ['resnet']
        },
        "epochs": {
            "values": [10]
        },
        "nf": {
            "values": [8, 16, 32]
        },
        "org":{
            "values": [1.5, 2]
        },
        "filter":{
            "values": [3, 2]
        },
        "activation": {
            "values": ['relu']
        },
        "batch_norm": {
            "values": [True, False]
        },
        "drop":{
            "values": [0.3, 0.4]
        },
        "dense":{
            "values": [64]
        },
        "lr": {
            "values": [0.001]
        },
        "optim":  {
            "values": ['adam']
        },
        "freeze": {
            "values": [10, 8, 7]
        }
    }
}

mode = sweep_config['parameters']['model']['values'][0]
size = 256 if mode == 'scratch' else 224

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
    wandb_logger = WandbLogger(log_model="all", project="FundDL-AS2")
    config = wandb.config
    if config.model == 'scratch':
        wandb.run.name = "nf_{}-{}x_f_{}_{}_{}_fc_{}_dr_{}_lr_{}_{}".format(config.nf, config.org, config.filter, config.activation, config.batch_norm, config.dense, config.drop, config.lr, config.optim)
        model = ScratchCNN(wandb.config)
    else:
        wandb.run.name = "resnet_lr_{}_{}_freeze_{}".format(config.lr, config.optim, config.freeze)
        model = ModdedResnet50(wandb.config)
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    trainer = pl.Trainer(max_epochs=config.epochs, precision=16, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    trainer.test(model=model, dataloaders=testloader)
    wandb.run.finish()

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train, count=1)