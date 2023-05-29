import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset

# I have chosen Tamil as the language
dataset_dir = '{}/{}/{}_'.format(os.getcwd(), 'tam', 'tam')
class TransliterationDataset(Dataset):
    def __init__(self, mode='train'):
        self.prepare_dataset(mode)
        
    def __len__(self):
        return len(self.enc_data)
    
    def __getitem__(self, idx):
        return self.enc_data[idx], self.dec_data[idx]
    
    def prepare_dataset(self, mode):
        df = pd.read_csv(dataset_dir+mode+'.csv')
        df = df.dropna(axis=0)

        iset, oset = set({' '}), set({' '})
        iseqLength, oseqLength = 0, 0

        for k, row in df.iterrows():
            # the model needs to know when the word starts and ends
            # \t is Start-Of-Sequence
            # \n is End-Of-Sequence
            # whitespace is padding token
            row[0] = '\t{}\n'.format(row[0])

            # Find the unique set of characters in both input and output
            for c in row[1]: iset.add(c)
            for c in row[0]: oset.add(c)

            iseqLength = max(iseqLength, len(row[1]))
            oseqLength = max(oseqLength, len(row[0]))

        # This dict holds index-to-char and char-to-index mappings of both input and output characters
        # The models will use this dict to encode/decode tokens
        # Also, it stores maximum input and output sequence lengths
        self.maps = {
            'ic2i': dict([(ichar, i) for i, ichar in enumerate(sorted(list(iset)))]),
            'i2ic': dict([(i, ichar) for i, ichar in enumerate(sorted(list(iset)))]),
            'oc2i': dict([(ochar, i) for i, ochar in enumerate(sorted(list(oset)))]),
            'i2oc': dict([(i, ochar) for i, ochar in enumerate(sorted(list(oset)))]),
            'iseq': iseqLength,
            'oseq': oseqLength
        }

        # The data is organized as [batch, sequence_length]
        # Tokens will be stored as their indices and not as one-hot vectors
        encoder_data = np.zeros((k+1, iseqLength), dtype='int64')
        decoder_data = np.zeros((k+1, oseqLength), dtype='int64')

        for k, row in df.iterrows():
            # Tokens at their locations in the sequences are stored as indices in the data
            # The rest of the sequence is padded by whitespace till its maximum length
            for t, c in enumerate(row[1]): encoder_data[k, t] = self.maps['ic2i'][c]
            encoder_data[k, t+1:] = self.maps['ic2i'][' ']
            for t, c in enumerate(row[0]): decoder_data[k, t] = self.maps['oc2i'][c]
            decoder_data[k, t+1:] = self.maps['oc2i'][' ']
        
        self.enc_data = encoder_data
        self.dec_data = decoder_data

        print(mode+'_dataset ', encoder_data.shape, decoder_data.shape)
     