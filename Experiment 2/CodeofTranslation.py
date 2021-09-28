# Language Translation
from __future__ import unicode_literals, print_function, division

from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import tkinter
from matplotlib import *

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.ticker as ticker

import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class language_set:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addingASentence(self, sentence):
        for word in sentence.split(' '):
            self.addingWord(word)

    def addingWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim
def NormalizeString(s):
     s = s.lower().strip()
     if ' ' not in s:
        s = list(s)
        s = ' '.join(s)
     s = unicodeToAscii(s)
     s = re.sub(r"([.!?])", r" \1", s)
     return s


def readLanguages(Language1, Language2, reverse=False):
    print("Reading lines...")
    Lines = open('%s-%s.txt' % (Language1, Language2), encoding='utf-8'). \
        read().strip().split('\n')
    Pairs = [[NormalizeString(s) for s in l.split('\t')[:2]] for l in Lines]


    # Reverse pairs
    if reverse:
        Pairs = [list(reversed(p)) for p in Pairs]
        Input_Language = language_set(Language2)
        Output_Language = language_set(Language1)

    else:

        Input_Language = language_set(Language1)
        Output_Language = language_set(Language2)

    return Input_Language, Output_Language, Pairs

MAX_LENGTH = 10

English_Prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def FilterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(English_Prefixes)


def FilterPairs(Pairs):
    return [pair for pair in Pairs if FilterPair(pair)]


def PrepareData(Language1, Language2, reverse=False):
    Input_Language, Output_Language, Pairs = readLanguages(Language1, Language2, reverse)
    print("Read %s sentence pairs" % len(Pairs))
    Pairs = FilterPairs(Pairs)
    print("Trimmed to %s sentence pairs" % len(Pairs))
    print("Counting words...")
    for pair in Pairs:
        Input_Language.addingASentence(pair[0])
        Output_Language.addingASentence(pair[1])
    print("Counted words:")
    print(Input_Language.name, Input_Language.n_words)
    print(Output_Language.name, Output_Language.n_words)
    return Input_Language, Output_Language, Pairs


Input_Language, Output_Language, Pairs = PrepareData('eng', 'cmn', True)
print(random.choice(Pairs))


class EncoderRNN(nn.Module):
    def __init__(self, Input_Size, Hidden_Size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = Hidden_Size
        self.embedding = nn.Embedding(Input_Size, Hidden_Size)
        self.gru = nn.GRU(Hidden_Size, Hidden_Size)

    def forward(self, Input, Hidden):
        Embedded = self.embedding(Input).view(1, 1, -1)
        Output = Embedded
        Output, Hidden = self.gru(Output, Hidden)
        return Output, Hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, Hidden_Size, Output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = Hidden_Size
        self.output_size = Output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, Input, Hidden, encoder_outputs):
        embedded = self.embedding(Input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], Hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                  encoder_outputs.unsqueeze(0))
        Output = torch.cat((embedded[0], attn_applied[0]), 1)
        Output = self.attn_combine(Output).unsqueeze(0)
        Output = F.relu(Output)
        Output, Hidden = self.gru(Output, Hidden)
        Output = F.log_softmax(self.out(Output[0]), dim=1)
        return Output, Hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def TimeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (-%s)' % (asMinutes(s), asMinutes(rs))


Teacher_Forcing_Ratio = 0.5

def Train(Input_tensor, Target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    Input_Length = Input_tensor.size(0)
    Target_Length = Target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(Input_Length):
        encoder_output, encoder_hidden = encoder(
            Input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    Use_TeacherForcing = True if random.random() < Teacher_Forcing_Ratio else False

    if Use_TeacherForcing:
        # FEED THE TARGET AS NEXT DATA
        for di in range(Target_Length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, Target_tensor[di])
            decoder_input = Target_tensor[di]

    else:
        # WITHOUT TEACHER FORCING
        for di in range(Target_Length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, Target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / Target_Length


def IndexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_fromSentence(lang, sentence):
    indexes = IndexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    Input_tensor = tensor_fromSentence(Input_Language, pair[0])
    Target_tensor = tensor_fromSentence(Output_Language, pair[1])
    return Input_tensor, Target_tensor


def TrainIter(encoder, decoder, n_iterators, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(Pairs))
                      for i in range(n_iterators)]
    criterion = nn.NLLLoss()

    for Iter in range(1, n_iterators + 1):

        Training_pair = training_pairs[Iter - 1]
        Input_tensor = Training_pair[0]
        Target_tensor = Training_pair[1]

        loss = Train(Input_tensor, Target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if Iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (TimeSince(start, Iter / n_iterators),
                                         Iter, Iter / n_iterators * 100, print_loss_avg))
        if Iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        Input_tensor = tensor_fromSentence(Input_Language, sentence)
        Input_length = Input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(Input_length):
            encoder_output, encoder_hidden = encoder(Input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(Output_Language.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(Pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print('')


hidden_size = 256
encoder1 = EncoderRNN(Input_Language.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, Output_Language.n_words, dropout_p=0.1).to(device)

TrainIter(encoder1, attn_decoder1, 2000, print_every=1000)

evaluateRandomly(encoder1, attn_decoder1)

def showAttention(input_sentence, output_words, attentions):

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    #ax.set_xticklabels([''] + input_sentence.split(' ') +
                       #['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("你 好 。 ")

