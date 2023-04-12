import torch
from torch.nn import Sequential, Linear, CrossEntropyLoss
from torch.optim import Adam, RMSprop
from torchmetrics.functional import accuracy
from torchvision.models import resnet50

import pytorch_lightning as pl

class ModdedResnet50(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Conv2D, BatchNorm, Relu, MaxPool, Blocks-1,2,3,4, AdaptiveAvgPool, Linear
        frame = resnet50(weights="DEFAULT")
        for l, layer in enumerate(frame.children()):
            for param in layer.parameters():
                param.requires_grad = False if (l < config.freeze) else True
        n_features = frame.fc.in_features
        fc_removed = list(frame.children())[:-1]

        self.resnet = Sequential(*fc_removed)
        self.dense = Linear(n_features, 10)

        if config.optim == 'adam':
          self.optim = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=config.lr)
        else:
          self.optim = RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=config.lr)
          
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        features = self.resnet(x).flatten(1)
        return self.dense(features)

    def configure_optimizers(self):
        return self.optim
    
    def training_step(self, batch, i):
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss
    
    def validation_step(self, batch, i):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        return preds
    
    def test_step(self, batch, i):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        return preds
    
    def _get_preds_loss_accuracy(self, batch):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        acc = accuracy(preds, y, task='multiclass', num_classes=10)
        return preds, loss, acc