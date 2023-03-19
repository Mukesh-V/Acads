import math
import random
import numpy as np
import seaborn as sns

from keras.datasets import fashion_mnist
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from activations import *
from data_utils import *
from nn_utils import *
from optimizers import *

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))/255.0
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))/255.0
X_test = X_test.T

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
X_train = X_train.T
X_val = X_val.T

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
ntrain = X_train.shape[1]
nval = X_val.shape[1]

def train(hparams):  
  # wandb.init(project="FundDL-AS1", config=hparams)
  # config = wandb.config
  # wandb.run.name = "{}_hl{}_bs_{}_ac_{}".format(config.loss, ", ".join(map(str, config.nn)), config.batch, config.activation)
  beta1 = 0.9
  beta2 = 0.999
  e = 1e-8
  gamma = 0.9
  config = hparams
  config['nn'].insert(0, X_train.shape[0])
  config['nn'].append(10)
  Wb, history, grads = nn_init(config['nn'], imode='xavier')

  correct_ones = 0
  for i in range(config['epoch']):
    loss, val_loss = 0, 0
    print("Epoch - ", i+1)
    for j in range(ntrain):
      X = np.reshape(X_train[:, j], (-1, 1))
      Y, Hs, As = forward_propagation(X, Wb, config['activation'])
      if y_train[j] == np.argmax(Y): correct_ones += 1
      if not (j+1) % config['batch']:
        grads_point = backpropagation(Wb, Y, [y_train[j]], config['activation'], config['decay'], config['loss'],Hs, As)
        for k in range(2):
            for l in range(len(Wb[0])):
                grads[k][l] = grads[k][l] + grads_point[k][l]
        # if config['optimizer'] == 'nadam':
        #   Wb_ahead, update = nadam(Wb, grads_epoch, config['eta'], eta, history, v)

fashion_hyperparams = {
    'nn': [128, 64, 32, 16],
    'decay': 0,
    'optimizer': 'nadam',
    'init': 'xavier',
    'activation': 'relu',
    'batch': 16,
    'eta': 0.001,
    'epoch': 20,
    'loss': 'mse'
}
train(fashion_hyperparams)