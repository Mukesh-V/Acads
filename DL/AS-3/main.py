from data import TransliterationDataset
from model import Transliterator

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

sweep_config = {
  "name": "Sweep-RNN",
  "method": "grid",
  "project": "FundDL-AS3",
  "metric":{
      "name":"val_accuracy",
      "goal":"maximize"
  },
  "parameters": {
        "model": {
            "values": ['no-attention']
        },
        "unit": {
            "values": ['gru']
        },
        "epochs": {
            "values": [20]
        },
        "embedding": {
            "values": [8]
        },
        "hidden": {
            "values": [8]
        },
        "layers": {
            "values": [1]
        },
        "drop": {
            "values": [0.05]
        },
        "optim":  {
            "values": ['adam']
        }
    }
}

batch_size = 512
trainloader = DataLoader(TransliterationDataset(), batch_size, True, num_workers=8)
valloader = DataLoader(TransliterationDataset('valid'), batch_size, num_workers=8)
testloader = DataLoader(TransliterationDataset('test'), batch_size, num_workers=8)

def train():
    wandb_logger = WandbLogger(log_model="all", project="FundDL-AS3")
    config = wandb.config
    if config.model == 'no-attention':
        wandb.run.name = "no-att_{}-{}x_{}-{}_rdr_{}_{}_{}".format(config.unit, config.layers, config.embedding, config.hidden, config.drop, config.optim, config.epochs)
        model = Transliterator(wandb.config, trainloader.dataset.maps)

    checkpoint_callback = ModelCheckpoint(monitor="val_seq_acc", mode="max")
    trainer = pl.Trainer(max_epochs=config.epochs, precision='bf16-mixed', logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    trainer.test(model=model, dataloaders=testloader)
    wandb.run.finish()

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train, count=1)