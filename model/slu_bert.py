# coding=utf8
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from TorchCRF import CRF
from transformers import (BertConfig, BertForMaskedLM,
                          BertForNextSentencePrediction, BertModel,
                          BertPreTrainedModel, BertTokenizer)

from utils.example import Example


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class JointBERT(BertPreTrainedModel):
    def __init__(self, config):
        
        
        model_name = 'bert-base-chinese'
        #MODEL_PATH = './chinese_wwm_ext_pytorch'
        # a.通过词典导入分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # b. 导入配置文件
        self.model_config = BertConfig.from_pretrained(model_name)
        # 修改配置
        self.model_config.output_hidden_states = True
        self.model_config.output_attentions = True
        super(JointBERT, self).__init__(self.model_config)
        # 通过配置和路径导入模型
        self.bert = BertModel(self.model_config)

        #self.args = args
        #self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = config.num_tags
        self.dropout_rate=0.1
        #self.bert = BertModel(config=config)  # Load pretrained bert
        self.slot_classifier = SlotClassifier(self.model_config.hidden_size, self.num_slot_labels, self.dropout_rate)

        self.O_token_id = Example.label_vocab.convert_tag_to_idx('O')

    def forward(self,batch):#, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        slot_labels_ids = batch.tag_ids
        attention_mask = batch.tag_mask.long()
        max_length = len(batch.input_ids[0])
        # input_ids = batch.input_ids.long()
        tokens = [self.tokenizer.tokenize(x) for x in batch.utt]
        input_ids = [self.tokenizer.convert_tokens_to_ids(["[CLS]"] + token + ["[SEP]"]) for token in tokens]
        input_ids = [x+[self.tokenizer.pad_token_id]*(max_length-len(x)) for x in input_ids]
        input_ids = torch.stack([torch.tensor(x) for x in input_ids])

        lengths = batch.lengths
        batch_size = len(batch)
        

        token_type_ids=torch.zeros(batch_size,max_length).long()

        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        #intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        #if intent_label_ids is not None:
        #    if self.num_intent_labels == 1:
        #        intent_loss_fct = nn.MSELoss()
        #        intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
        #    else:
        #        intent_loss_fct = nn.CrossEntropyLoss()
        #        intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
        #    total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += slot_loss

        #outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here
        #outputs = (total_loss,) + outputs
        #return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
        
        outputs = (slot_logits,) 
        outputs = outputs + (total_loss,)
        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
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

class JointBERT_slotonly(nn.Module):
    def __init__(self, config):
        super(JointBERT_slotonly, self).__init__()
        model_name = 'bert-base-chinese'
        MODEL_PATH = './chinese_wwm_ext_pytorch'
        # a.通过词典导入分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # b. 导入配置文件
        model_config = BertConfig.from_pretrained(model_name)
        # 修改配置
        model_config.output_hidden_states = True
        model_config.output_attentions = True
        # 通过配置和路径导入模型
        self.bert_model = BertModel.from_pretrained(model_name)


        self.config = config
        n_layers=2
        dropout=0.1
        self.word_embed = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=0)
        self.encoder = EncoderRNN(config.vocab_size, config.hidden_size, n_layers, dropout=dropout)
        self.decoder = LuongAttnDecoderRNN('dot', config.hidden_size, config.num_tags, n_layers, dropout=dropout)
        self.loss_fct = nn.NLLLoss(ignore_index=config.tag_pad_idx)
        self.O_token_id = Example.label_vocab.convert_tag_to_idx('O')

    def forward(self, batch, teacher_forcing_ratio=0.5):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        batch_size = len(batch)

        max_length = MAX_LENGTH

        input_ids = torch.transpose(input_ids, 0, 1)
        tag_ids = torch.transpose(tag_ids, 0, 1)
        input_length = input_ids.size()[0]
        encoder_outputs, encoder_hidden = self.encoder(input_ids, lengths, None)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([self.O_token_id] * batch_size))
        decoder_hidden = encoder_hidden[:self.decoder.n_layers] # Use last (forward) hidden state from encoder

        all_decoder_outputs = Variable(torch.zeros(input_length, batch_size, self.decoder.output_size))

        outputs=[]
        # Run through decoder one time step at a time
        for t in range(input_length):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            all_decoder_outputs[t] = decoder_output
            decoder_input = tag_ids[t] # Next input is current target
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            outputs.append(decoder_output)
        # Loss calculation and backpropagation
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
            tag_ids.transpose(0, 1).contiguous(), # -> batch x seq
            lengths
        )
        ##########################
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

class SLUSeq2SeqTagging(nn.Module):

    def __init__(self, config):
        super(SLUSeq2SeqTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size //
                                          2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(
            config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)
        packed_inputs = rnn_utils.pack_padded_sequence(
            embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(
            packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(
            packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
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
                value = ''.join([batch.utt[i][j] for j in idx_buff])
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
