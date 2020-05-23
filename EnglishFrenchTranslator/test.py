# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:58:26 2020

@author: Brian
"""
import sys,time,math


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


import ParallelSequenceDataset as ds
from Model import EncoderRNN, DecoderRNN
from Language import SOS_token,EOS_token


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODELS_DIR = "../../Models"
MODEL_NAME = "model"
PATH = f"{MODELS_DIR}/{MODEL_NAME}.pth"
checkpoint = torch.load(PATH)

MAX_LENGTH = 10
hidden_size = 256
data = checkpoint['data']

enc = EncoderRNN(data.source_language.n_words, hidden_size, device).to(device)
dec = DecoderRNN(hidden_size, data.target_language.n_words, device).to(device)

enc.load_state_dict(checkpoint['encoder_state_dict'])
dec.load_state_dict(checkpoint['decoder_state_dict'])





def evaluate(encoder, decoder, sequence_tensor, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = sequence_tensor
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        #decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            #decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(data.target_language.idx2word[topi.item()])
            
            #print(decoder_output.data.topk(3))
            decoder_input = topi.squeeze().detach()

        return decoded_words#, decoder_attentions[:di + 1]


sequence = "i am very happy with you"
sequence_tensor = data.source_language.tensor_from_sequence(sequence).to(device)

output_sequence = evaluate(enc,dec,sequence_tensor)
print(output_sequence)






