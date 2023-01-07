#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

### Multi-Head Attention ###

from math import sqrt
 
class MultiHeadAttention(nn.Module):
    dim_input: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
 
    def __init__(self, dim_input, dim_k, dim_v, num_heads=4):
        super(MultiHeadAttention, self).__init__()

        #维度必须能被num_head 整除
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_input = dim_input
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads

        #定义线性变换矩阵
        self.linear_q = nn.Linear(dim_input, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_input, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_input, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
 
    def forward(self, x):
        # x: tensor of shape (batch, n, dim_input)
        batch, n, dim_input = x.shape
        assert dim_input == self.dim_input
 
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
 
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
 
        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
 
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att

############################




class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

        ### Multi-Head Attention ###
        self.attention = MultiHeadAttention(config.hidden_size, config.hidden_size, config.hidden_size)
        self.attention_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_norm_2 = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.att_fnn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        #############################



    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)

        ### Because we delete the sort in batch.py ###
        ### Modify here __[enforce_sorted=False]__ ###
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        
        ### Multi-Head Attention ###
        norm1_rnn_out = self.attention_norm_1(rnn_out)
        norm2_rnn_out = self.attention_norm_2(rnn_out)
        # print('rnn_out_origin:', rnn_out)
        # print('norm1_rnn_out:', norm1_rnn_out)
        # print('norm2_rnn_out:', norm2_rnn_out)
        rnn_out = self.attention(norm1_rnn_out)
        # rnn_out = self.attention(norm2_rnn_out)
        # print('rnn_out_modify:', rnn_out)
        ############################
        
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
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob
