import numpy as np
import json
import collections

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.bert_word2vec import BertUtils
from utils.evaluator import Evaluator


class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()

        cls.word_vocab = Vocab(padding=True, unk=True,BOS_and_EOS=True, filepath=train_path)
        cls.label_vocab = LabelVocab(root)

        cls.word2vec = Word2vecUtils(word2vec_path)
        #cls.word2vec = BertUtils()

    @classmethod
    def load_dataset(cls, data_path, BOS_and_EOS=False):
        datas = json.load(open(data_path, 'r', encoding='utf-8'),
                          object_pairs_hook=collections.OrderedDict)
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt, BOS_and_EOS)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, BOS_and_EOS=False):
        super(Example, self).__init__()
        self.ex = ex

        ### for test.py ###
        self.id = ex['utt_id']
        self.pred = []
        ###################

        self.utt = ex['asr_1best']
        self.slot = {}

        ### for test.py ###
        if 'semantic' in ex:
            labelList = ex['semantic']
        elif 'pred' in ex:
            labelList = ex['pred']
        for label in labelList:
            ###################
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot,
                          value in self.slot.items()]
        if BOS_and_EOS:
            self.input_idx = [Example.word_vocab['<s>']] + [Example.word_vocab[c]
                                                            for c in self.utt] + [Example.word_vocab['</s>']]
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(
                'O')]+[l.convert_tag_to_idx(tag) for tag in self.tags]+[l.convert_tag_to_idx('O')]
        else:
            self.input_idx = [Example.word_vocab[c] for c in self.utt]
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
