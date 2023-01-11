# coding=utf8
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

from utils.example import Example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 50

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,embed=None, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding_size = embed_size
        self.embedding=embed
        #self.embedding = nn.Embedding(input_size, hidden_size)
        #self.embedding.weight.data.copy_(torch.eye(hidden_size))
        #self.embedding.weight.requires_grad = False

        #self.gru = nn.GRU(hidden_size, hidden_size)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(1, batch_size, self.embedding_size)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.embedding.weight.data.copy_(torch.eye(hidden_size))
        #self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        output = self.embedding(input).view(1, batch_size, self.hidden_size)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=0)
        self.encoder = EncoderRNN(
            config.vocab_size, config.embed_size, config.hidden_size, self.word_embed).to(device)
        #self.encoder = EncoderRNN(
        #    config.vocab_size, config.hidden_size).to(device)
        self.decoder = DecoderRNN(
            config.hidden_size, config.num_tags).to(device)
        self.output_layer = TaggingFNNDecoder(
            config.hidden_size, config.num_tags, config.tag_pad_idx)
        self.loss_fct = nn.NLLLoss(ignore_index=config.tag_pad_idx)
        self.O_token_id = Example.label_vocab.convert_tag_to_idx('O')

    def forward(self, batch, teacher_forcing_ratio=0.7):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        batch_size = len(batch)

        max_length = input_ids.size()[1]

        input_ids = torch.transpose(input_ids, 0, 1)
        tag_ids = torch.transpose(tag_ids, 0, 1)

        encoder_hidden = self.encoder.initHidden(batch_size)
        input_length = input_ids.size()[0]
        target_length = tag_ids.size()[0]
        encoder_outputs = Variable(torch.zeros(
            max_length, batch_size, self.encoder.hidden_size, device=device))
        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_ids[ei], batch_size,encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = Variable(torch.LongTensor([self.O_token_id] * batch_size))
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(input_length):
                #decoder_output, decoder_hidden, decoder_attention = self.decoder(
                #    decoder_input, decoder_hidden, encoder_outputs)
                decoder_output, decoder_hidden= self.decoder(
                    decoder_input, batch_size, decoder_hidden)
                loss += self.loss_fct(decoder_output, tag_ids[di])
                decoder_input = tag_ids[di]  # Teacher forcing
                outputs.append(decoder_output)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(input_length):
                #decoder_output, decoder_hidden, decoder_attention = self.decoder(
                #    decoder_input, decoder_hidden, encoder_outputs)
                decoder_output, decoder_hidden= self.decoder(
                    decoder_input, batch_size, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.loss_fct(decoder_output, tag_ids[di])
                # if decoder_input.item() == Example.label_vocab.convert_tag_to_idx:
                #    break
                outputs.append(decoder_output)
        outputs=torch.stack(outputs)
        outputs=torch.transpose(outputs,0,1)
        return outputs, loss

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch, teacher_forcing_ratio=0)
        #prob = torch.stack(prob)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])+1]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j-1] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j-1] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(
                logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob
