import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sys


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_width, stride=1, pool_size=0):

        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_width, stride=1)
        self.act = nn.ReLU()
        self.pool_size = pool_size

        if pool_size > 0:
            self.pool = nn.MaxPool1d(self.pool_size, self.pool_size)

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = self.act(x)

        if self.pool_size > 0:
            x = self.pool(x)

        return x

class Net(nn.Module):
    def __init__(self, num_layers=1, input_channels=30, window_size=12000):

        super(Net, self).__init__()
        self.input_channels = input_channels
        self.window_size = window_size

        self.conv1 = ConvBlock(in_channels=self.input_channels, out_channels=70, kernel_width=17, stride=1, pool_size=4)
        self.do1 = nn.Dropout(p = .1)
        self.conv2 = ConvBlock(in_channels=70, out_channels=90, kernel_width=7, stride=1, pool_size=4)
        self.do2 = nn.Dropout(p = .1)
        self.conv3 = ConvBlock(in_channels=90, out_channels=70, kernel_width=5, stride=1, pool_size=4)
        self.do3 = nn.Dropout(p = .1)
        self.conv4 = ConvBlock(in_channels=70, out_channels=20, kernel_width=5, stride=1)
        self.pool = nn.AdaptiveMaxPool1d(900//20)
        self.do4 = nn.Dropout(p = .1)
  
        self.rnn1 = nn.LSTM(input_size=900, hidden_size=2400, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=4800, hidden_size=2400, num_layers=num_layers, batch_first=True, bidirectional=True)
        # self.rnn3 = nn.LSTM(input_size=4800, hidden_size=1800, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(4800, 1200)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(1200, 200) #CHANGED 100 -> 200
        # self.act2 = nn.ReLU()

        # self.rnn1 = nn.LSTM(input_size=900, hidden_size=1800, num_layers=num_layers, batch_first=True, bidirectional=True)
        # self.rnn2 = nn.LSTM(input_size=3600, hidden_size=1800, num_layers=num_layers, batch_first=True, bidirectional=True)
        # self.rnn3 = nn.LSTM(input_size=3600, hidden_size=1800, num_layers=num_layers, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(3600, 900)
        # self.act = nn.ReLU()
        # self.fc2 = nn.Linear(900, 200) #CHANGED 100 -> 200
        # self.act2 = nn.ReLU()

    def forward(self, x, hidden_state=None, seq_length=200):

        assert x.shape[0] == self.input_channels
        # print(x.shape)
        x = torch.as_strided(x, (seq_length, self.input_channels, self.window_size), (50, x.shape[1], 1))
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.do1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.do2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = self.do3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = self.pool(x)
        x = self.do4(x)
        # print(x.shape)
        x = x.view(1, seq_length, x.shape[1]*x.shape[2])  
        # print(x.shape)
        res1, hidden_state = self.rnn1(x, None)
        res2, hidden_state = self.rnn2(res1, None)
        res2 = res2 + res1
        # res3, hidden_state = self.rnn3(res2, None)
        x = self.fc(res2) # + res3)
        x = self.act(x)
        x = self.fc2(x)
        # x = self.act2(x)
        return x, hidden_state

    def loss(self, prediction, label, seq_length = 200, reduction='mean'):
        l2_loss = F.mse_loss(prediction.view(-1, seq_length, 200), label.view(-1, seq_length, 200), reduction=reduction)
        return l2_loss


class CNN(nn.Module):
    def __init__(self, num_layers=1, input_channels=30, window_size=12000):

        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.window_size = window_size

        self.conv1 = ConvBlock(in_channels=self.input_channels, out_channels=70, kernel_width=17, stride=1, pool_size=4)
        self.do1 = nn.Dropout(p = .1)
        self.conv2 = ConvBlock(in_channels=70, out_channels=90, kernel_width=7, stride=1, pool_size=4)
        self.do2 = nn.Dropout(p = .1)
        self.conv3 = ConvBlock(in_channels=90, out_channels=70, kernel_width=5, stride=1, pool_size=4)
        self.do3 = nn.Dropout(p = .1)
        self.conv4 = ConvBlock(in_channels=70, out_channels=20, kernel_width=5, stride=1, pool_size=0)
        self.pool = nn.AdaptiveMaxPool1d(900//20)
        self.do4 = nn.Dropout(p = .1)
  
        self.fc1 = nn.Linear(900,900)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(900, 100)

    def forward(self, x, hidden_state=None, seq_length=200):

        assert x.shape[0] == self.input_channels

        x = torch.as_strided(x, (seq_length, self.input_channels, self.window_size), (100, x.shape[1], 1))
        x = self.conv1(x)
        x = self.do1(x)
        x = self.conv2(x)
        x = self.do2(x)
        x = self.conv3(x)
        x = self.do3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.do4(x)
        
        x = x.view(seq_length, x.shape[1]*x.shape[2])  
        # print(x.shape)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x, 0

    def loss(self, prediction, label, seq_length = 200, reduction='mean'):
        
        l2_loss = F.mse_loss(prediction.view(-1, seq_length, 100), label.view(-1, seq_length, 100), reduction=reduction)
        return l2_loss 



class Disc(nn.Module):
    
    def __init__(self):

        super(Disc, self).__init__()

        self.conv1 = nn.Conv2d(1, 25, 5, stride=1, padding=2)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(25, 25, 5, stride=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(25, 25, 5, stride=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(25, 10, 5, stride=1)
        self.act4 = nn.ReLU()
        # self.pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(5400, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.act4(x)
        # x = self.pool4(x)
        x = x.view(1, -1)
        x = self.fc1(x)

        return x

    def loss(self, prediction, label, reduction='mean'):
        
        loss_val = F.binary_cross_entropy_with_logits(prediction, label, reduction='mean')
        return loss_val










