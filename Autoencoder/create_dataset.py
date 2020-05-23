# -*- coding: utf-8 -*-
"""
Created on Sat May 23 02:38:58 2020

@author: Brian
"""
import SequenceUtils as seq_utils
filename = "eng-fra"


# Read the file and split into lines
lines = open('data/%s.txt' % (filename), encoding='utf-8').\
    read().strip().split('\n')
    

pairs = [[seq_utils.normalizeString(s) for s in l.split('\t')] for l in lines]


with open("data/autoencoder_data.txt", "x") as file:
    for i in range(0,len(pairs)):
        print(i)
        file.write(f"{pairs[i][0]}\t{pairs[i][0]}\n")