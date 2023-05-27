from data import TransliterationDataset
from model_no_att import Transliterator

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
wandb.login()

sweep_config = {
  "name": "Sweep-RNN",
  "method": "grid",
  "project": "FundDL-AS3",
  "metric":{
      "name":"val_seq_acc",
      "goal":"maximize"
  },
  "parameters": {
        "model": {
            "values": ['no-attention']
        },
        "unit": {
            "values": ['rnn']
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
            "values": [2]
        },
        "drop": {
            "values": [0.05]
        },
        "bi":{
            "values": [True]
        }
    }
}

batch_size = 512
trainloader = DataLoader(TransliterationDataset(), batch_size, True, num_workers=2)
valloader = DataLoader(TransliterationDataset('valid'), batch_size, num_workers=2)

def train():
    wandb_logger = WandbLogger(log_model="all", project="FundDL-AS3")
    config = wandb.config
    if config.model == 'no-attention':
        bi_status = 'Bi' if config.bi else 'Uni'
        wandb.run.name = "no-att-lev_{}-{}x-{}_{}-{}_rdr_{}_{}".format(config.unit, config.layers, bi_status, config.embedding, config.hidden, config.drop, config.epochs)
        model = Transliterator(wandb.config, trainloader.dataset.maps)

    checkpoint_callback = ModelCheckpoint(monitor="val_seq_acc", mode="max")
    trainer = pl.Trainer(max_epochs=config.epochs, precision=16, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    wandb.run.finish()

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train, count=10)