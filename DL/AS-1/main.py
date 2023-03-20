import math
import numpy as np

from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split

from activations import *
from data_utils import *
from nn_utils import *
from optimizers import *

import wandb
wandb.login()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-wp', '--wandb_project', dest='proj', type=str, help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', dest='name', type=str, help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-d', '--dataset', dest='dataset', type=str)
parser.add_argument('-e', '--epochs', dest='epoch', type=int, help='Number of epochs to train neural network.')
parser.add_argument('-b', '--batch_size', dest='batch', type=int, help='Batch size used to train neural network.')
parser.add_argument('-l', '--loss', dest='loss', type=str)
parser.add_argument('-o', '--optimizer', dest='optimizer', type=str)
parser.add_argument('-lr', '--learning_rate', dest='eta', type=float, help='Learning rate used to optimize model parameters')
parser.add_argument('-m', '--momentum', dest='gamma', type=float, help='Momentum used by momentum and nag optimizers.')
parser.add_argument('-beta', '--beta', dest='beta', type=float, help='Beta to be used by rmsprop, adam and nadam')
parser.add_argument('-beta2', '--beta2', dest='beta2', type=float, help='Beta2 to be used by adam and nadam')
parser.add_argument('-eps', '--epsilon', dest='e', type=float)
parser.add_argument('-w_d', '--weight_decay', dest='decay', type=float)
parser.add_argument('-w_i', '--weight_init', dest='init', type=str)
parser.add_argument('-a', '--activation', dest='activation', type=str)
parser.add_argument('-hl','--hidden', dest='nn', nargs='+', type=int)
args = parser.parse_args()

project = args.proj if args.proj else "FundDL-AS1"
experiment = "mnist" if args.dataset == "mnist" else "fashion-mnist"

beta1 = 0.9 if not args.beta else args.beta
beta2 = 0.999 if not args.beta2 else args.beta2
e = 1e-8 if not args.e else args.e
gamma = 0.9 if not args.gamma else args.gamma

def train():  
  wandb.init(project="FundDL-AS1")
  config = wandb.config
  wandb.run.name = "{}_{}_hl{}_bs_{}_ac_{}".format(experiment, config.loss, ", ".join(map(str, config.nn)), config.batch, config.activation)

  config.nn.insert(0, X_train.shape[0])
  config.nn.append(10)
  Wb, history, grads = nn_init(config.nn, imode='xavier')
  v = history.copy()

  for i in range(config.epoch):
    loss, val_loss = 0, 0
    correct_ones = 0
    print("Epoch - ", i+1)
    for j in range(ntrain):
      X = np.reshape(X_train[:, j], (-1, 1))
      Y, Hs, As = forward_propagation(X, Wb, config.activation)
      if y_train[j] == np.argmax(Y): correct_ones += 1
      grads_point = backpropagation(Wb, Y, [y_train[j]], config.activation, config.decay, config.loss, Hs, As)
      if config.optimizer == 'sgd':
        Wb = gd(Wb, grads_point, config.eta)
      else:
        for k in range(2):
            for l in range(len(Wb[0])):
                grads[k][l] += grads_point[k][l]
        if not (j+1) % config.batch:
          for k in range(2):
            for l in range(len(Wb[0])):
              grads[k][l] /= config.batch 
          
          if config.optimizer == 'momentum':
            Wb, history = momentum(Wb, grads, config.eta, gamma, history)
          elif config.optimizer == 'rmsprop':
            Wb, history = rmsprop(Wb, grads, config.eta, history, beta1, e)
          elif config.optimizer == 'adam':
            Wb, history, v = adam(Wb, grads, config.eta, history, v, beta1, beta2, e, i+1)
          elif config.optimizer == 'nadam':
            Wb, history, v = nadam(Wb, grads, config.eta, history, v, beta1, beta2, e, i+1)
        
        if config.loss == 'cross_entropy':
          loss -= (1/ntrain) * math.log(Y[y_train[j]])
        elif config.loss == 'mse':
          loss += (1/ntrain) * (np.argmax(Y) - y_train[j])**2

    y_hat_val, _, _ = forward_propagation(X_val, Wb, config.activation)
    count_val = np.sum(np.argmax(y_hat_val, axis = 0)== y_val)

    for j in range(nval):
      X_v= np.reshape(X_val[:,j], (-1, 1)) 
      y_hat_val, _, _ = forward_propagation(X_v, Wb, config.activation)

      if config.loss == 'cross_entropy':
        val_loss = val_loss - (1/nval)*math.log(y_hat_val[int(y_val[j])])
      elif config.loss == 'mse':
        val_loss = val_loss + (1/nval)*(np.argmax(y_hat_val) - y_val[j])**2

    accuracy = 100*correct_ones/ntrain
    val_accuracy = 100*count_val/nval

    print("Loss:", loss)
    print("Accuracy:",accuracy)
    print("Validation Loss:", val_loss)
    print("Validation Accuracy:", val_accuracy)

    metrics = {'epoch':i, 'val_accuracy': val_accuracy, 'val_loss': val_loss, 'accuracy': accuracy, 'loss': loss}
    wandb.log(metrics)

  Y_test, _, _ = forward_propagation(X_test, Wb, config.activation)
  wandb.log({"Confusion_Matrix" : wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_test,
                        preds=np.argmax(Y_test, axis = 0),
                        class_names=labels)})
  wandb.run.finish()

if experiment == 'fashion-mnist':
  (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
  labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
else: 
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  labels = list(range(10))

# for i in range(10):
#   wandb.init(project="FundDL-AS1")
#   wandb.run.name = "Sample-Images"
#   for j in range(30):
#     if y_train[j] == i:
#       wandb.log({"examples": [wandb.Image(X_train[j], caption=labels[i])]})
#       break

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))/255.0
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))/255.0
X_test = X_test.T

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
X_train = X_train.T
X_val = X_val.T

ntrain = X_train.shape[1]
nval = X_val.shape[1]

sweep_config = {
  "name": "Sweep-CE",
  "method": "grid",
  "project": "FundDL-AS1",
  "metric":{
      "name":"val_accuracy",
      "goal":"maximize"
  },
  "parameters": {
        "epoch": {
            "values": [20]
        },
        "nn": {
            "values":[[64]]
        },
        "decay":{
            "values":[0]
        },
        "eta":{
            "values":[0.001]
        },
        "batch": {
            "values":[32]
        },  
        "optimizer": {
            "values":['sgd']
        },
        "init": {
            "values":['xavier']
        },
        "activation":{
            "values": ['relu']
        },
        "loss":{
            "values": ['cross_entropy']
        }
    }
}
args = vars(args)
for param in ['epoch', 'nn', 'decay', 'eta', 'batch', 'optimizer', 'init', 'activation', 'loss']:
  if args[param]:
    sweep_config['parameters'][param]['values'] = [args[param]]
sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train, count=1)