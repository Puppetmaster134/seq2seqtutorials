# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:14:39 2020

@author: Brian
"""

import torch

SOS_token = 0
EOS_token = 1


class Language:
    def __init__(self):
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def sequence_to_indices(self, sequence):
        return [self.word2idx[word] for word in sequence.split(' ')]
    
    def tensor_from_sequence(self,sequence):
        indices = self.sequence_to_indices(sequence)
        indices.append(EOS_token)
        return torch.tensor(indices, dtype=torch.long).view(-1, 1)
