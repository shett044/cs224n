#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1e
class CNN(nn.Module):
    def __init__(self, embd_char_size, embd_word_len, k=5):
        super(CNN, self).__init__()
        self.cnn_m = nn.Conv1d(embd_char_size, embd_word_len, k)

    def forward(self, x_reshape):
        # print("CNN forward()")
        # print("x_reshape => ", self.x_reshape.size())
        cnn_res = F.relu(self.cnn_m(x_reshape))
        # print("cnn_res => ", cnn_res.size())
        max_pool_filter = cnn_res.size()[-1]
        x_conv_out = nn.MaxPool1d(max_pool_filter)(cnn_res)
        # print("x_conv_out => ", x_conv_out.size())

        return x_conv_out

### END YOUR CODE
