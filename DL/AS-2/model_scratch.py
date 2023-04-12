import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, MaxPool2d, Flatten, Dropout, Linear
from torch.nn import ReLU, GELU, SiLU, Mish
from torch.nn import LogSoftmax, CrossEntropyLoss
from torch.optim import Adam, RMSprop
from torchmetrics.functional import accuracy
import pytorch_lightning as pl

class ScratchCNN(pl.LightningModule):
    def get_output_shape(self, model, image_dim):
      return model(torch.rand(*(image_dim))).data.shape

    def __init__(self, config):
        super().__init__()
        if config.activation == 'relu':
            self.activ = ReLU()
        elif config.activation == 'gelu':
            self.activ = GELU()
        elif config.activation == 'silu':
            self.activ = SiLU()
        elif config.activation == 'mish':
            self.activ = Mish()

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
            Linear(config.dense, 10),
            LogSoftmax()
        )

        eta = config.lr
        if config.optim == 'adam':
            self.optim = Adam(self.parameters(), lr=eta)
        if config.optim == 'rmsprop':
            self.optim = RMSprop(self.parameters(), lr=eta)

    def forward(self, x):
        features = self.conv(x)
        y_hat = self.dense(features)
        return y_hat
    
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