#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1d
class Highway(nn.Module):
    """
    Works like a skip connection in ResNet but defines gates for the convolution result
    """

    def __init__(self, inp_size, hidden_size):
        """

        :param conv_out_tor: convolution output of character embedding
        :param dropout_rate: Drop out rate (default to 0.2)
        """
        super(Highway, self).__init__()
        self.embed_size = inp_size
        self.proj_NN = nn.Linear(inp_size, hidden_size, bias=True)
        self.gate_NN = nn.Linear(inp_size, hidden_size, bias=True)

    def forward(self, x_conv_out):
        """
        Feed forward neural network with dropout layer
        x_proj involves RELU that will be carried forward but its value is gated by x_gate that weight how much information to carry forward
        :return: x_highway: tensor of batch_size, word_embed_size
        """
        # print("Highway: forward method")
        # print("Size of x_conv_out: ", self.x_conv_out.size())
        x_proj = F.relu(self.proj_NN(x_conv_out))
        x_gate = nn.Sigmoid()(self.gate_NN(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out

        # print("Size of x_word_embd: ", x_word_embd.size())
        return x_highway

### END YOUR CODE
