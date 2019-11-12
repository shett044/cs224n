#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


#
# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab, dropout_rate=0.3):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()


        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        self.ember_char_size = 50
        self.embedding = nn.Embedding(len(vocab.char2id), self.ember_char_size, padding_idx=vocab.char2id[vocab.PAD_CHAR])
        self.cnn = CNN(self.ember_char_size, self.embed_size)
        self.highway = Highway(self.embed_size, self.embed_size)
        self.dropout = nn.Dropout(dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        # print("Embedding forward()")
        ### YOUR CODE HERE for part 1f
        # print("Size of input => ", input.size())
        x_emb = self.embedding(input)
        # print("Size of embed_res => ", x_emb.size())
        x_reshape = x_emb.permute(0, 1, 3, 2)
        sent, batch, char, word = x_reshape.size()
        x_reshape = x_reshape.view(-1, char, word)
        c_res = self.cnn(x_reshape)
        c_res = c_res.reshape(sent, batch, -1)
        x_highway = self.highway(c_res)
        x_word_embd = self.dropout(x_highway)
        return x_word_embd
        ### END YOUR CODE
