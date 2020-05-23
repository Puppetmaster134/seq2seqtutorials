# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:28:42 2020

@author: Brian
"""
import sys
import torch
from torch.utils.data import Dataset

from Language import Language, SOS_token, EOS_token
import SequenceUtils as seq_utils


class ParallelSequenceDataset(Dataset):

    def __init__(self, filename, max_len=10, device='cpu'):
        self.max_len = max_len
        source_language,target_language,pairs = self.read_file(filename)
        
        self.device = device
        
        self.source_language = source_language
        self.target_language = target_language
        self.sequence_pairs = pairs
        
        
    def __len__(self):
        return len(self.sequence_pairs)

    def __getitem__(self, idx):
        pair = self.sequence_pairs[idx]
        
        #print(pair[0])
        #print(pair[1])
        
        source_indices = self.source_language.sequence_to_indices(pair[0])
        target_indices = self.target_language.sequence_to_indices(pair[1])
        
        source_indices.append(EOS_token)
        target_indices.append(EOS_token)
        
        source_tensor = torch.tensor(source_indices, dtype=torch.long, device=self.device).view(-1, 1)
        target_tensor = torch.tensor(target_indices, dtype=torch.long, device=self.device).view(-1, 1)
        
        return (source_tensor,target_tensor)
    
    def filter_sequence_pairs(self,pairs):
        
        eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )
        
        
        
        return [pair for pair in pairs if \
                len(pair[0].split(' ')) < self.max_len and \
                len(pair[1].split(' ')) < self.max_len and \
                pair[0].startswith(eng_prefixes)]
    
    def is_valid_pair(self, source, target):
        eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )
        
        return len(source.split(' ')) < self.max_len and \
                len(target.split(' ')) < self.max_len and \
                source.startswith(eng_prefixes)
        
        



    def filterPair(self,p):
        eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )
                
        return len(p[0].split(' ')) < self.max_len and \
            len(p[1].split(' ')) < self.max_len and \
            p[0].startswith(eng_prefixes)
    
    
    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]
    
    def read_file(self, filename):
        print("Reading lines...")
    
        # Read the file and split into lines
        lines = open('data/%s.txt' % (filename), encoding='utf-8').\
            read().strip().split('\n')
    
        # Split every line into pairs and normalize
        pairs = [[seq_utils.normalizeString(s) for s in l.split('\t')][:2] for l in lines]
        
        pairs = self.filterPairs(pairs)
        
        source = Language()
        target = Language()
        
        for pair in pairs:
            if self.is_valid_pair(pair[0],pair[1]):
                source.addSentence(pair[0])
                target.addSentence(pair[1])
        
        print(f'Source language counted words: {source.n_words}')
        print(f'Target language counted words: {target.n_words}')
        
        return source,target,pairs