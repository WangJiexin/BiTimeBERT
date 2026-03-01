import math
from tqdm import tqdm
import numpy as np
import torch
from math import sqrt, log
from . import data_utils, FairseqDataset
from itertools import chain
import json
import random
from collections import defaultdict

class ParagraphInfo(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def get_word_piece_map(self, sentence):
        return [self.dictionary.is_start_word(i) for i in sentence]

    def get_word_at_k(self, sentence, left, right, k, word_piece_map=None):
        num_words = 0
        while num_words < k and right < len(sentence):
            # complete current word
            left = right
            right = self.get_word_end(sentence, right, word_piece_map)
            num_words += 1
        return left, right

    def get_word_start(self, sentence, anchor, word_piece_map=None):
        word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
        left  = anchor
        while left > 0 and word_piece_map[left] == False:
            left -= 1
        return left
    # word end is next word start
    def get_word_end(self, sentence, anchor, word_piece_map=None):
        word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
        right = anchor + 1
        while right < len(sentence) and word_piece_map[right] == False:
            right += 1
        return right

class MaskingScheme:
    def __init__(self, args):
        self.args = args
        self.mask_ratio = 0.15

    def mask(tokens, tagmap=None):
        pass

class BertRandomMaskingScheme(MaskingScheme):
    def __init__(self, args, tokens, pad, mask_id):
        super().__init__(args)
        self.pad = pad
        self.tokens = tokens
        self.mask_id = mask_id

    def mask(self, sentence, temp_infor_list, temp_masked_ratio, tagmap=None):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.mask_ratio)

        mask_temp_num = math.ceil(len(temp_infor_list)*temp_masked_ratio)
        mask_temp_idx_list = random.sample(range(len(temp_infor_list)), mask_temp_num)
        mask = np.array([], dtype=int)
        not_mask = np.array([], dtype=int)
        for temp_i, temp_infor in enumerate(temp_infor_list):
            b_sent_pos, e_sent_pos, b_temp_pos, e_temp_pos, temp_ids, temp_grad = temp_infor
            if temp_i in mask_temp_idx_list:
                if mask_num-(e_temp_pos-b_temp_pos)<0:
                    not_mask = np.append(not_mask, np.array(range(b_temp_pos, e_temp_pos)))
                    continue
                else:
                    mask = np.append(mask, np.array(range(b_temp_pos, e_temp_pos)))
                    mask_num = mask_num-(e_temp_pos-b_temp_pos)
            else:
                not_mask = np.append(not_mask, np.array(range(b_temp_pos, e_temp_pos)))

        candidate_mask = np.array(range(sent_length))
        candidate_mask = candidate_mask[~np.isin(candidate_mask,mask)]
        candidate_mask = candidate_mask[~np.isin(candidate_mask,not_mask)]
        if len(candidate_mask)==0:
            sent_length = len(sentence)
            mask_num = math.ceil(sent_length * self.mask_ratio)
            mask = np.random.choice(sent_length, mask_num, replace=False)
            return bert_masking(sentence, mask, self.tokens, self.pad, self.mask_id)
        else:
            non_temp_mask = np.random.choice(candidate_mask, size=mask_num, replace=False)
            mask = np.append(mask, non_temp_mask)
            return bert_masking(sentence, mask, self.tokens, self.pad, self.mask_id)

def bert_masking(sentence, mask, tokens, pad, mask_id):
    sentence = np.copy(sentence)
    sent_length = len(sentence)
    target = np.copy(sentence)
    mask = set(mask)
    for i in range(sent_length):
        if i in mask:
            rand = np.random.random()
            if rand < 0.8:
                sentence[i] = mask_id
            elif rand < 0.9:
                # sample random token according to input distribution
                sentence[i] = np.random.choice(tokens)
        else:
            target[i] = pad
    return sentence, target, None
