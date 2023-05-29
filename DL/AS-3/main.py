from data import TransliterationDataset
from model_no_att import Transliterator
from model_att import AttentionTransliterator

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
wandb.login()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tf', dest='teacher_force', type=float)
parser.add_argument('-l', '--lr', dest='lr', type=float)

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
            "values": ['attention']
        },
        "unit": {
            "values": ['gru']
        },
        "epochs": {
            "values": [30]
        },
        "embedding": {
            "values": [16]
        },
        "hidden": {
            "values": [64]
        },
        "layers": {
            "values": [3]
        },
        "drop": {
            "values": [0.05]
        }
    }
}

batch_size = 512
trainloader = DataLoader(TransliterationDataset(), batch_size, True, num_workers=2)
valloader = DataLoader(TransliterationDataset('valid'), batch_size, num_workers=2)

args = parser.parse_args()
eta = 0.005 if not args.lr else args.lr
tf = 0.5 if not args.teacher_force else args.teacher_force
# Higher this value, less will be the teacher forcing
# One idea is that this can be made dynamic as a function of epochs
# so that at higher epochs, teacher forcing is reduced
def train():
    wandb_logger = WandbLogger(log_model="all", project="FundDL-AS3")
    config = wandb.config
    if config.model == 'no-attention':
        wandb.run.name = "no-att-lev-lr-{}_{}-{}x_{}-{}_rdr_{}_{}".format(eta, config.unit, config.layers, config.embedding, config.hidden, config.drop, config.epochs)
        
        # Passing the maps to the model to figure out input, output sizes
        model = Transliterator(wandb.config, trainloader.dataset.maps, eta, tf)
    else:
        wandb.run.name = "att-lev-lr-{}_{}-{}x_{}-{}_rdr_{}_{}".format(eta, config.unit, config.layers, config.embedding, config.hidden, config.drop, config.epochs)
        model = AttentionTransliterator(wandb.config, trainloader.dataset.maps, eta, tf)

    # Store models in Artifact Registry of wandb
    checkpoint_callback = ModelCheckpoint(monitor="val_seq_acc", mode="max")
    trainer = pl.Trainer(max_epochs=config.epochs, precision=16, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    wandb.run.finish()

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train, count=1)