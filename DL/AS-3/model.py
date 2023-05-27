import torch
import random
import numpy as np
import Levenshtein
from torch.nn import Embedding, RNN, GRU, LSTM, Linear
from torch.nn import Softmax, CrossEntropyLoss
from torch.optim import Adam
import pytorch_lightning as pl

eta = 0.01
teacher_force = 0.5
class Transliterator(pl.LightningModule):
    def __init__(self, config, maps):
        super().__init__()
        self.config = config

        if config.unit == 'rnn':
            self.unit = RNN
        elif config.unit == 'gru':
            self.unit = GRU
        else:
            self.unit = LSTM

        self.maps = maps

        self.enc_embedding = Embedding(len(self.maps['ic2i']), config.embedding)
        self.encoder = self.unit(input_size=config.embedding, hidden_size=config.hidden, num_layers=config.layers, dropout=config.drop, batch_first=True)
        self.dec_embedding = Embedding(len(self.maps['oc2i']), config.embedding)
        self.decoder = self.unit(input_size=config.embedding, hidden_size=config.hidden, num_layers=config.layers, dropout=config.drop, batch_first=True)
        self.fc = Linear(config.hidden, len(self.maps['oc2i']))

        self.loss = CrossEntropyLoss()

    def forward(self, x, y=None):
        e_embed = self.enc_embedding(x)
        _, e_hidden = self.encoder(e_embed)

        d_input = np.zeros((x.shape[0], 1), dtype='int64')
        d_input[:, 0] = self.maps['oc2i']['\t']
        d_input = torch.from_numpy(d_input)

        outputs, d_hidden = [], e_hidden
        for i in range(self.maps['oseq']):
            if random.random() > teacher_force and y is not None: 
                d_input = y[:, i].unsqueeze(1)
            d_embed = self.dec_embedding(d_input)
            d_output, d_hidden = self.decoder(d_embed, d_hidden)
            fc_output = self.fc(d_output)

            d_input = torch.argmax(Softmax(dim=2)(fc_output), dim=2)
            outputs.append(fc_output)
        
        op = torch.stack(outputs, dim=1)
        return torch.squeeze(op)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=eta)
    
    def training_step(self, batch, i):
        x, y = batch
        logits = self(x, y).view(-1, len(self.maps['i2oc']))
        y = y.view(-1)
        loss = self.loss(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, i):
        cacc, sacc = self._get_accuracy(batch)
        self.log('val_char_acc', cacc, prog_bar=True)
        self.log('val_seq_acc', sacc, prog_bar=True)
        return cacc, sacc
    
    def _get_accuracy(self, batch):
        x, y = batch
        preds = torch.argmax(Softmax(dim=2)(self(x)), dim=2)

        char_acc, seq_acc = 0.0, 0.0
        for pred, truth in zip(preds, y):
            truth_word = ''.join([self.maps['i2oc'][i.item()] for i in truth]).replace('\t', '').replace('\n', '').replace(' ', '')
            pred_word = ''.join([self.maps['i2oc'][i.item()] for i in pred]).replace('\t', '').replace('\n', '').replace(' ', '')
            
            correct_chars = 0
            for pred_token, truth_token in zip(pred_word, truth_word):
                if pred_token == truth_token: correct_chars += 1
            char_acc += correct_chars/len(truth_word)

            distance = Levenshtein.distance(pred_word, truth_word)
            seq_acc += (1 - distance/(max(len(truth_word), len(pred_word))))

        char_acc /= len(x)
        seq_acc /= len(x)
        return char_acc, seq_acc
    