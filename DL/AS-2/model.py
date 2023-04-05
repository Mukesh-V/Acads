from torch.nn import Sequential, Conv2d, ReLU, Linear, LogSoftmax, MaxPool2d, BatchNorm2d, Flatten
from torch.optim import Adam, NAdam, SGD, RMSprop

class CNN:
    def __init__(self):
        self.model = None

    def model(self, config):
        layers = []

        if config.activation == 'relu':
            activ = ReLU()

        for i in range(4):
            layers.append(Conv2d(int(config.filters * (config.org**i)), int(config.filters * (config.org**i+1)), config.kernel))
            layers.append(activ)
            if config.batch_norm:
                layers.append(BatchNorm2d(int(config.filters * (config.org**i+1))))
            layers.append(MaxPool2d(config.pool))

            layers.append(Flatten())
            for i in range(len(config.dense)-1):
                layers.append(Linear(config.dense[0]), Linear(config.dense[1]))
                layers.append(activ)
            layers.append(Linear(config.dense[-1], 10))
            layers.append(LogSoftmax())

        model = Sequential(*layers)
        self.model = model
        
        return model
