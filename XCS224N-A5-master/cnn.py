#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1e
class CNN(nn.Module):
    def __init__(self, x_reshape, embd_word_len, k=5):
        super(CNN, self).__init__()
        self.k = k
        self.x_reshape = x_reshape
        self.embd_word_len = embd_word_len
        self.sentence_len, self.batch_size, self.embd_char_len, self.word_len = x_reshape.size()

    def forward(self):
        # print("CNN forward()")
        self.x_reshape = self.x_reshape.view(-1, self.embd_char_len, self.word_len)
        # print("x_reshape => ", self.x_reshape.size())
        cnn_m = nn.Conv1d(self.embd_char_len, self.embd_word_len, self.k)
        cnn_res = F.relu(cnn_m(self.x_reshape))
        # print("cnn_res => ", cnn_res.size())
        max_pool_filter = cnn_res.size()[-1]
        x_conv_out = nn.MaxPool1d(max_pool_filter)(cnn_res)
        x_conv_out = x_conv_out.reshape(self.sentence_len, self.batch_size, -1)
        # print("x_conv_out => ", x_conv_out.size())

        return x_conv_out

### END YOUR CODE
