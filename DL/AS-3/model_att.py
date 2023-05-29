import torch
import random
import numpy as np
import Levenshtein
from torch.nn import Embedding, RNN, GRU, Linear
from torch.nn import Softmax, CrossEntropyLoss
from torch.optim import Adam
import pytorch_lightning as pl

# Additive Attention has proven good for sequence generators
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

        # Embedding dict size is the number of output characters
        self.embedding = Embedding(len(maps['oc2i']), config.embedding)
        # This input size helps in accommodating the attended input (embedded input + context)
        self.decoder = self.unit(config.embedding + config.hidden, config.hidden, num_layers=config.layers, dropout=config.drop, batch_first=True)
        # Final output layer
        self.fc = Linear(config.hidden, len(maps['oc2i']))

    def forward(self, d_input, d_hidden, e_output):
        # Additive Attention
        query = self.query(d_hidden[-1]).unsqueeze(1)
        key = self.key(e_output)
        # This is where the "Addition" happens
        energy = torch.tanh(query + key)
        attention_weights = torch.softmax(self.energy(energy), dim=1)
        
        # Batch matrix multiplication is helpful in this case
        context = torch.bmm(attention_weights.transpose(1, 2), e_output)
        
        embedded = self.embedding(d_input)
        # Attended input is passed onto the rnn unit : just a 
        # concatenation of embedded input and the attention context
        attended_input = torch.cat((embedded, context), dim=2)
        d_output, d_hidden = self.decoder(attended_input, d_hidden)
        
        fc_output = self.fc(d_output)
        return fc_output, d_hidden
    
    
class AttentionTransliterator(pl.LightningModule):
    def __init__(self, config, maps, lr, tf):
        super().__init__()
        self.config = config

        if config.unit == 'rnn':
            self.unit = RNN
        else:
            self.unit = GRU

        self.maps = maps
        self.eta = lr
        self.teacher_force = tf

        # Encoder stays simple
        self.enc_embedding = Embedding(len(maps['ic2i']), config.embedding)
        self.encoder = self.unit(input_size=config.embedding, hidden_size=config.hidden, num_layers=config.layers, dropout=config.drop, batch_first=True)
        self.attention_decoder = AttentionDecoder(config, maps)

        self.loss = CrossEntropyLoss()

    def forward(self, x, y=None):
        e_embed = self.enc_embedding(x)
        e_output, e_hidden = self.encoder(e_embed)

        # The first input to the decoder will be a SOS token
        # which is \t in this work
        d_input = np.zeros((x.shape[0], 1), dtype='int64')
        d_input[:, 0] = self.maps['oc2i']['\t']

        # this single .to() caused me a whole two days of trouble
        # the absence of .to() forced me to run the code in cpu
        d_input = torch.from_numpy(d_input).to(device=self.device)

        outputs = []
        d_hidden = e_hidden
        for i in range(self.maps['oseq']):

            # "Flip a coin" and decide if needed to teacher-force
            # else feed the previous output
            if random.random() > self.teacher_force and y is not None: 
                d_input = y[:, i].unsqueeze(1)

            # Encoder outputs are required for Additive Attention
            d_output, d_hidden = self.attention_decoder(d_input, d_hidden, e_output)

            # Output will be [batch, 1, len(output characters)]
            # The following reduces it to [batch, 1], which is infact, 
            # expected as the input
            # Basically, it returns the token with maximum probability
            d_input = torch.argmax(Softmax(dim=2)(d_output), dim=2)
            outputs.append(d_output)
        
        # All the individual outputs are stacked along the sequence dimension
        # and squeezed to make it [batch, seq_length, len(output_characters)]
        op = torch.stack(outputs, dim=1)
        return torch.squeeze(op)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.eta)
    
    def training_step(self, batch, i):
        x, y = batch

        # CrossEntropy requires the vectors to be [group, num_classes]  
        # (the outputs should be one-hot vectors) and labels to be [group, ]
        # Here, groups = batches * seq_length
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
        # Outputs of the model are reduced to easier shapes
        preds = torch.argmax(Softmax(dim=2)(self(x)), dim=2)

        char_acc, seq_acc = 0.0, 0.0
        for pred, truth in zip(preds, y):
            # Generate the predicted and truth words, remove the SOS, EOS and space
            truth_word = ''.join([self.maps['i2oc'][i.item()] for i in truth]).replace('\t', '').replace('\n', '').replace(' ', '')
            pred_word = ''.join([self.maps['i2oc'][i.item()] for i in pred]).replace('\t', '').replace('\n', '').replace(' ', '')
            
            correct_chars = 0
            for pred_token, truth_token in zip(pred_word, truth_word):
                if pred_token == truth_token: correct_chars += 1
            char_acc += correct_chars/len(truth_word)

            # Levenshtein distance between two strings is the minimal 
            # number of edits required to make them equal
            # This is an attempt to not punish the model 
            # for predicting "carr" instead of "car"
            distance = Levenshtein.distance(pred_word, truth_word)
            seq_acc += (1 - distance/(max(len(truth_word), len(pred_word))))

        char_acc /= len(x)
        seq_acc /= len(x)
        return char_acc, seq_acc
    