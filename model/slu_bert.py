# coding=utf8
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
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

        outputs = (slot_logits,) 
        outputs = outputs + (total_loss,)
        return outputs  

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
