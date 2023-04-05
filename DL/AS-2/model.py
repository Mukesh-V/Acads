from torch.nn import Sequential, Conv2d, ReLU, BatchNorm2d, MaxPool2d, Flatten, Dropout, Linear, LogSoftmax, CrossEntropyLoss
from torch.optim import Adam
from torchmetrics.functional import accuracy
import torch
import pytorch_lightning as pl

class CNN(pl.LightningModule):
    def get_output_shape(self, model, image_dim):
      return model(torch.rand(*(image_dim))).data.shape

    def __init__(self, config):
        super().__init__()
        if config.activation == 'relu':
            self.activ = ReLU()

        self.criterion = CrossEntropyLoss()

        conv_layers = []
        for i in range(4):
            if i != 0:
              conv_layers.append(Conv2d(int(config.nf * (config.org**(i-1))), int(config.nf * (config.org**i)), kernel_size=config.filter))
            else:
              conv_layers.append(Conv2d(3, config.nf, kernel_size=config.filter))

            conv_layers.append(self.activ)

            if config.batch_norm:
              if i != 0:
                conv_layers.append(BatchNorm2d(int(config.nf * (config.org**i))))
              else:
                conv_layers.append(BatchNorm2d(config.nf))

            conv_layers.append(MaxPool2d(2))
        conv_layers.append(Flatten())
        conv_layers.append(Dropout(config.drop))
        self.conv = Sequential(*conv_layers)

        inp_dense = self.get_output_shape(self.conv, [1, 3, 256, 256])[1]

        self.dense = Sequential(
            Linear(inp_dense, config.dense),
            ReLU(),
            Dropout(config.drop),
            Linear(config.dense, 10),
            LogSoftmax()
        )

        self.save_hyperparameters()

    def forward(self, x):
        features = self.conv(x)
        y_hat = self.dense(features)
        return y_hat
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
    
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