from time import sleep # 
from matplotlib import pyplot
import torch
import pytorch_lightning as pl
import torchmetrics as tm

from torch import nn, optim
from torch.nn.utils import weight_norm  # TCN
from einops.layers.torch import Rearrange

from .utilities import OneHotEncode, PredictionEncoderForMCML
from .render import plot_metrics, plot_prediction

import numpy as np

class NNetModule(pl.LightningModule):
    def __init__(self, model: str,
                 n_inputs: int, # features freq bins* num of ratios
                 n_outputs: int, #  classes in 'one hot encoding'
                 n_steps: int,   # seq len
                 learning_rate: float, win_size: float, gpu: str):
        super().__init__()   

        # params
        self.name = model
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.learning_rate = learning_rate

        self._n_labels = 4          # Max nb. of speakers
        self._threshold = 0.5       # 
        self.win_size = win_size

        # loss & its transform
        self.criterion = nn.BCEWithLogitsLoss() #Binary cross entropy (logits: output of neurons before activation function. for numerical stability)
        self.tfy = OneHotEncode(n_outputs)

        # metrics & its transform
        average = 'micro'   
        self.fn_accuracy = tm.Accuracy(task="multilabel", num_labels=n_outputs,
                multidim_average='global', average=average)
        self.fn_precision = tm.Precision(task="multilabel", num_labels=n_outputs,
                multidim_average='global', average=average)
        self.fn_recall = tm.Recall(task="multilabel", num_labels=n_outputs,
                multidim_average='global', average=average)
        self.fn_cmatrix = tm.ConfusionMatrix(task="multilabel", num_labels=n_outputs,
                multidim_average='global', average=average, normalize='true')
        
        self.tfp = PredictionEncoderForMCML(
            n_outputs, self._n_labels, threshold=self._threshold) # convert predictions to indx

        # model
        if model == 'fnn':
            self.model = FnnClassifier(n_inputs*n_steps, n_outputs)
        elif model == 'rnn':
            self.model = RnnClassifier(n_inputs, n_steps, n_outputs)
        elif model == 'tcn':
            self.model = TCNModel(n_inputs, n_steps, n_outputs)
        else:
            raise ValueError("Unknown model inside NNetModule!")

        # Log Hyper-parameters
        self.save_hyperparameters()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """ Expect features as (B, T, ...) and reshpae based on model. """
        logits = self.model(features)

        return logits

    def configure_optimizers(self):
        """ Set the optimizer and LR scheduler. """
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, amsgrad=False
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 10, 45], gamma=0.2, last_epoch=-1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> float:
        """ Train on a batch and return loss for optimizer. """
        features, targets = batch
        predictions = self.forward(features)

        loss = self.criterion(predictions, self.tfy(targets).float())
        self.log('tr_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        features, targets = batch
        predictions = self.forward(features)

        loss = self.criterion(predictions, self.tfy(targets).float())
        self.log('val_loss', loss)

    # def validation_epoch_end(self, outputs):
    #     """ Sleep to allow GPU to cool down. """
    #     sleep(5)
        
    def test_step(self, batch, batch_idx) -> None:
        features, targets = batch
        predictions = self.forward(features)
        enc_pred = self.tfp(predictions, targets)

        # Compute Metrics ## not loss
        acc = self.fn_accuracy(predictions, self.tfy(targets))
        prec = self.fn_precision(predictions, self.tfy(targets))
        rec = self.fn_recall(predictions, self.tfy(targets))
        cmat = self.fn_cmatrix(predictions, self.tfy(targets))

        # Log Metrics (objects not values)
        self.log('te_accuracy', self.fn_accuracy, on_step=False, on_epoch=True)
        self.log('te_precision', self.fn_precision, on_step=False, on_epoch=True)
        self.log('te_recall', self.fn_recall, on_step=False, on_epoch=True)
        
        # Prediction plot
        plot_prediction(self.tfy(targets), torch.sigmoid(predictions), # sigmoid: to make the prediction probabilites
                     self.tfy(enc_pred),
                     {'A': acc, 'P': prec, 'R': rec, 'CM': cmat}, 
                     win_size=self.win_size,
                     model_name=self.name, 
                     log_dir=self.logger.log_dir)
    
    def test_epoch_end(self, outputs) -> None:
        
        plot_metrics({
            'Accuracy': self.fn_accuracy.compute(),
            'Precision': self.fn_precision.compute(),
            'Recall': self.fn_recall.compute(),
            'ConfusionMatrix': self.fn_cmatrix.compute()
        }, 'confusion_matrix.png', self.logger.log_dir, RT=0.2)


class FnnClassifier(nn.Module):
    """ Self-normalized FNN classifier. """

    def __init__(self, n_inputs, n_outputs):
        """ Setup the model (expect input of correct size). """
        super().__init__()

        # params
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.n_hidden = 200
        self.n_layers = 4
        self.dropout = 0.2

        # Model
        self.tfx = Rearrange('B T F R->B (T F R)') #B: batch, T: seq-len, F: freq bins, R: #Mic -1
                                                   # T is flatten 

        self.fnn = nn.Sequential(
            nn.Linear(n_inputs, self.n_hidden),
            nn.SELU(),
            nn.AlphaDropout(self.dropout),

            nn.Linear(self.n_hidden, self.n_hidden),
            nn.SELU(),
            nn.AlphaDropout(self.dropout),

            nn.Linear(self.n_hidden, self.n_hidden),
            nn.SELU(),
            nn.AlphaDropout(self.dropout),

            nn.Linear(self.n_hidden, self.n_hidden),
            nn.SELU(),
            nn.AlphaDropout(self.dropout),
        )

        self.fc = nn.Linear(self.n_hidden, n_outputs) # decoder can be attached to last layer of the encoder (all are FNN), its kept here to be simialr to other models

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """ Accepts features as (B, T, ...) and return logits as (B, C). """
        features = self.tfx(features) # reshaping
        embedding = self.fnn(features) # call the model (encoder)
        logits = self.fc(embedding)   # return logits (decoder)

        return logits


class RnnClassifier(nn.Module):
    """ An RNN base model for DOA classification. """

    def __init__(self, n_inputs: int, n_steps: int, n_outputs: int):
        """ Setup the model (expect inputs of correct size). """
        super().__init__()

        # params
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        self.n_outputs = n_outputs

        self.n_hidden = 500
        self.n_layers = 2
        self.dropout = 0.2

        # Model
        self.tfx = Rearrange('B T F R->T B (F R)')

        self.rnn = nn.GRU(n_inputs, self.n_hidden, self.n_layers,
                          batch_first=False, dropout=self.dropout) ## change RNN or GRU (hard coded)

        self.fc = nn.Linear(self.n_hidden, n_outputs) # decoder to match  the size from hidden to output

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """ 
        """
        batch_size = len(features)
        hidden = torch.zeros(self.n_layers, batch_size, self.n_hidden) # initilize the hidden state
        hidden = hidden.type_as(features) # move the hidden tensor to the same GPU as features

        #
        features = self.tfx(features)
        embedding, hidden = self.rnn(features, hidden)
        logits = self.fc(hidden[-1])       # many to one (current time= last row)

        return logits   #]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x) #


class TCNModel(nn.Module):

    # def __init__(self, num_channels, kernel_size=2, dropout=0.2):
    def __init__(self, n_inputs: int, n_steps: int, n_outputs: int):
        super(TCNModel, self).__init__()
        # from rnn
        self.n_inputs = n_inputs    # not used (need in_channels)
        self.n_steps = n_steps      # not used (maybe for out_channel?)
        self.n_outputs = n_outputs

        # from tcn
        self.n_hidden = 500  # channels for each block
        self.n_layers = 2
        self.dropout = 0.2
        self.kernel_size = 3

        # Model
        self.tfx = Rearrange('B T F R-> B (F R) T')

        self.tcn = TemporalConvNet(
            self.n_inputs, [self.n_hidden]*self.n_layers, kernel_size=self.kernel_size, dropout=self.dropout)
        self.decoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.n_hidden, self.n_outputs)
        )

    def forward(self, x):
        # preprocess
        batch_size = len(x)
        features = self.tfx(x)

        # apply NN
        embedding = self.tcn(features)[:, :, -1] # current time 
        logits = self.decoder(embedding)
        # print(x.shape, features.shape, embedding.shape, logits.shape)
        # input('/')

        return logits
