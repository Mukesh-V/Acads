import torch
import random
import numpy as np
import Levenshtein
from torch.nn import Embedding, RNN, GRU, Linear
from torch.nn import Softmax, CrossEntropyLoss
from torch.optim import Adam
import pytorch_lightning as pl

eta = 0.01
teacher_force = 0.5

class AttentionDecoder(pl.LightningModule):
    def __init__(self, config, maps):
        super().__init__()
        self.config = config

        if config.unit == 'rnn':
            self.unit = RNN
        else:
            self.unit = GRU

        self.query = Linear(config.hidden, config.hidden)
        self.key = Linear(config.hidden, config.hidden)
        self.energy = Linear(config.hidden, 1)

        self.embedding = Embedding(len(maps['oc2i']), config.embedding)
        self.decoder = self.unit(config.hidden*2, config.hidden, num_layers=config.layers, dropout=config.drop, batch_first=True)
        self.fc = Linear(config.hidden, len(maps['oc2i']))

    def forward(self, d_input, d_hidden, e_output):
        query = self.query(d_hidden[-1]).unsqueeze(1)
        key = self.key(e_output)
        energy = torch.tanh(query + key)
        attention_weights = torch.softmax(self.energy(energy), dim=1)
        
        context = torch.bmm(attention_weights.transpose(1, 2), e_output)
        
        embedded = self.embedding(d_input)
        attended_input = torch.cat((embedded, context), dim=2)
        d_output, d_hidden = self.decoder(attended_input, d_hidden)
        
        fc_output = self.fc(d_output)
        return fc_output, d_hidden
    
    
class AttentionTransliterator(pl.LightningModule):
    def __init__(self, config, maps):
        super().__init__()
        self.config = config

        if config.unit == 'rnn':
            self.unit = RNN
        else:
            self.unit = GRU

        self.maps = maps

        self.enc_embedding = Embedding(len(maps['ic2i']), config.embedding)
        self.encoder = self.unit(input_size=config.embedding, hidden_size=config.hidden, num_layers=config.layers, dropout=config.drop, batch_first=True)
        self.attention_decoder = AttentionDecoder(config, maps)

        self.loss = CrossEntropyLoss()

    def forward(self, x, y=None):
        e_embed = self.enc_embedding(x)
        e_output, e_hidden = self.encoder(e_embed)

        d_input = np.zeros((x.shape[0], 1), dtype='int64')
        d_input[:, 0] = self.maps['oc2i']['\t']
        d_input = torch.from_numpy(d_input)

        outputs = []
        d_hidden = e_hidden
        for i in range(self.maps['oseq']):
            if random.random() > teacher_force and y is not None: 
                d_input = y[:, i].unsqueeze(1)
            d_output, d_hidden = self.attention_decoder(d_input, d_hidden, e_output)
            d_input = torch.argmax(Softmax(dim=2)(d_output), dim=2)
            outputs.append(d_output)
        
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
    