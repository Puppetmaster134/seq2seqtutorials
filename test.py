# -*- coding: utf-8 -*-
"""
Created on Tue May 19 00:38:43 2020

@author: Brian
"""
import sys,time,math,random


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

MAX_LENGTH = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = ds.ParallelSequenceDataset("eng-fra", max_len=MAX_LENGTH, device=device)
loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)




hidden_size = 256
enc = EncoderRNN(data.source_language.n_words, hidden_size, device).to(device)
dec = DecoderRNN(hidden_size, data.target_language.n_words, device).to(device)



def train(source_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10):
    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    source_tensor = source_tensor.view(-1,1)
    target_tensor = target_tensor.view(-1,1)
    
    source_length = source_tensor.size(0)
    target_length = target_tensor.size(0)
    
    #Encoder output at each step. These will be attended to while decoding
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
        
    for source_idx in range(source_length):
        encoder_output, encoder_hidden = encoder(source_tensor.view(-1,1)[source_idx], encoder_hidden)
        encoder_outputs[source_idx] = encoder_output[0,0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    
    
    
    loss = 0
    teacher_forcing_ratio = 0.5
    #use_teacher_forcing = True
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for target_idx in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[target_idx])
            decoder_input = target_tensor[target_idx]  # Teacher forcing
    else:
        # Teacher forcing: Feed the target as the next input
        for target_idx in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = target_tensor[target_idx]  # Teacher forcing
            
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[target_idx])
            if decoder_input.item() == EOS_token:
                break
    
            
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
    
    



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("mygraph_autoenc.png")


def run_training(dataloader,encoder,decoder, num_epochs, print_every_percent=10, plot_every_percent=10, learning_rate=0.01):
    start = time.time()
    plot_losses = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    
    print_divisor = int(len(dataloader) / print_every_percent)
    plot_divisor = int(len(dataloader) / plot_every_percent)
    
    total_count = 0
    training_total = len(dataloader) * num_epochs
    for epoch in range(0,num_epochs):
        count = 0
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        
        for source,target in dataloader:
            loss = train(source,target,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,10)
            count+=1
            total_count += 1
            
            
            print_loss_total += loss
            plot_loss_total += loss
            
            if count % print_divisor == 0:
                percent = total_count / training_total
                print_loss_avg = print_loss_total / print_divisor
                print_loss_total = 0
                print('Epoch %d -- %s (%d %d%%) %.4f' % (epoch + 1,timeSince(start, percent),total_count, percent * 100, print_loss_avg))
                
            if count % plot_divisor == 0:
                plot_loss_avg = plot_loss_total / plot_divisor
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    
    print(plot_losses)
    showPlot(plot_losses)



run_training(loader,enc,dec,20)

PATH = "model_autoenc.pth"

torch.save({
            'encoder_state_dict': enc.state_dict(),
            'decoder_state_dict': dec.state_dict(),
            'data':data
            }, PATH)





























