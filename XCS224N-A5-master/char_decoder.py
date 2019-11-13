#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        print("char_embedding_size, hidden_size: ", char_embedding_size, hidden_size)
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        # TODO: check this target_vocab.char2id
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab

        ### END YOUR CODE

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        # print(input.size())
        x_emb = self.decoderCharEmb(input.contiguous())
        dec_res, dec_hidden = self.charDecoder(x_emb, dec_hidden)
        # print(dec_res.size())
        scores = self.char_output_projection(dec_res)
        return scores, dec_hidden

        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        loss_func = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')

        inp = char_sequence[:-1, :]
        out = char_sequence[1:, :]
        # print("Inp %s and output %s" % (inp.size(), out.size()))
        tgt = out.reshape(-1)

        scores, dec_hidden = self.forward(inp, dec_hidden)
        # print("tgt ", tgt.size())
        # print("scores ", scores.size())
        scores = scores.reshape(-1, 30)
        return loss_func(scores, tgt)
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        start_t = torch.tensor([self.target_vocab.start_of_word], dtype=torch.long, device=device)
        batch_size = initialStates[0].size(1)
        start_t = start_t.expand(batch_size, 1).transpose(1,0)
        print(start_t)
        print(start_t.size())
        start_emb = self.decoderCharEmb(start_t)

        ht, ct = initialStates
        res_word = torch.zeros((1, batch_size,), dtype=torch.long, device=device)
        softmax = nn.Softmax(dim=1)
        for i in range(max_length):
            res, (ht, ct) = self.charDecoder(start_emb, (ht, ct))
            scores = self.char_output_projection(res)
            print(scores)
            print(scores.size())
            next_char = softmax(scores).argmax(2)
            start_t = next_char
            print(start_t.size())
            start_emb = self.decoderCharEmb(start_t)

            res_word = torch.cat((res_word, start_t), 0)

        res_batch = res_word.reshape(5, -1).tolist()
        res_list = [[self.target_vocab.id2char[x] for x in res[1:]] for res in res_batch]

        final_words = []
        for s in res_list:
            word = ''
            i = 0
            len_s = len(s)
            while i < len_s and s[i] != "}":
                word += s[i]
                i += 1
            final_words.append(word)
        return final_words

        ### END YOUR CODE
