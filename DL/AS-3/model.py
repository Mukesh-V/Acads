import torch
import numpy as np
from torch.nn import Embedding, RNN, GRU, LSTM, Linear
from torch.nn import LogSoftmax, NLLLoss, Softmax
from torch.optim import Adam, RMSprop
import pytorch_lightning as pl

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

        self.loss = NLLLoss()
        
        eta = 0.01
        if config.optim == 'adam':
            self.optim = Adam(self.parameters(), lr=eta)
        if config.optim == 'rmsprop':
            self.optim = RMSprop(self.parameters(), lr=eta)

    def forward(self, x, y):
        e_embed = self.enc_embedding(x)
        _, e_hidden = self.encoder(e_embed)

        d_embed = self.dec_embedding(y)
        d_output, _ = self.decoder(d_embed, e_hidden)
        output = LogSoftmax(dim=2)(self.fc(d_output))
        return output
    
    def infer(self, x):
        e_embed = self.enc_embedding(x.unsqueeze(0))
        _, e_hidden = self.encoder(e_embed)

        flag = False
        decoded_word, decoded_char = "", "\t"
        d_hidden = e_hidden
        while not flag:
            target = np.array([[self.maps['oc2i'][decoded_char]]], dtype='int64')
            target = torch.from_numpy(target)

            d_embed = self.dec_embedding(target)
            d_output, d_hidden = self.decoder(d_embed, d_hidden)
            fc_output = LogSoftmax(dim=2)(self.fc(d_output))

            output = torch.argmax(fc_output, dim=2)
            decoded_char = self.maps['i2oc'][output.item()]
            decoded_word += decoded_char
            if decoded_char == '\n' or len(decoded_word) > 30:
                flag = True

        return decoded_word
    
    def configure_optimizers(self):
        return self.optim
    
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
    
    def test_step(self, batch, i):
        cacc, sacc = self._get_accuracy(batch)
        self.log('test_char_acc', cacc, prog_bar=True)
        self.log('test_seq_acc', sacc, prog_bar=True)
        return cacc, sacc
    
    def _get_accuracy(self, batch):
        x, y = batch

        char_acc, seq_acc = 0.0, 0.0
        for i, truth in enumerate(y):
            truth_word = ''.join([self.maps['i2oc'][i.item()] for i in truth]).replace('\t', '').replace('\n', '').replace(' ', '')
            pred_word = self.infer(x[i]).replace('\t', '').replace('\n', '').replace(' ', '')
            
            correct_chars = 0
            for pred_token, truth_token in zip(pred_word, truth_word):
                if pred_token == truth_token: correct_chars += 1
            char_acc += correct_chars/len(truth_word)

            if truth_word == pred_word: 
                seq_acc += 1 

        char_acc /= len(x)
        seq_acc /= len(x)
        return char_acc, seq_acc
    