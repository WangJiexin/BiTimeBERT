# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

import iso8601
from datetime import datetime, timedelta
from babel.dates import format_date, format_datetime, format_time
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer

import torch
from torch.utils.data import ConcatDataset
from fairseq.data import Dictionary, IndexedInMemoryDataset, IndexedRawTextDataset,data_utils
from fairseq.data.temp_bert_dataset import BlockPairDataset, TempBertDataset
from fairseq.data.no_nsp_temp_bert_dataset import BlockDataset, NoNSPTempBertDataset
from . import FairseqTask, register_task
from fairseq.data.masking import ParagraphInfo
from bitarray import bitarray

class BertDictionary(Dictionary):
    """Dictionary for BERT tasks
        extended from Dictionary by adding support for cls as well as mask symbols"""
    def __init__(
        self,
        pad='[PAD]',
        unk='[UNK]',
        cls='[CLS]',
        mask='[MASK]',
        sep='[SEP]'
    ):
        super().__init__(pad, unk)
        (
            self.cls_word,
            self.mask_word,
            self.sep_word,
        ) = cls, mask, sep
        self.is_start = None
        self.nspecial = len(self.symbols)

    def class_positive(self):
        return self.cls()

    def cls(self):
        """Helper to get index of cls symbol"""
        idx = self.add_symbol(self.cls_word)
        return idx

    def mask(self):
        """Helper to get index of mask symbol"""
        idx = self.add_symbol(self.mask_word)
        return idx

    def sep(self):
        """Helper to get index of sep symbol"""
        idx = self.add_symbol(self.sep_word)
        return idx

    def is_start_word(self, idx):
        if self.is_start is None:
            self.is_start = [not self.symbols[i].startswith('##') for i in range(len(self))]
        return self.is_start[idx]

@register_task('temp_bert')
class TempBertTask(FairseqTask):
    """
    Train BERT model.
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """
    def __init__(self, args, dictionary):
        super().__init__(args)  #set self.datasets={}, self.args = args
        self.dictionary = dictionary
        #self.dictionary.symbols->list, self.dictionary.indices->dict
        args.vocab_size = len(dictionary)         #28996
        self.seed = args.seed                     
        self.no_nsp = args.no_nsp                 #no_nsp=False/True
        self.short_seq_prob = 0.0                 #short_seq_prob=0.0
        self.date_prediction_list = self.return_date_prediction_list()
        self.date2tempexp = self.return_date2tempexp_dict(args.temp_granularity)
        self.date2tokenids_dict = self.return_date2tokenids_dict()
        self.date2idx_dict = self.return_date2idx_dict(args.temp_granularity)
        self.date_num = self.date2idx_dict["2007-06-19"]+1
    @property
    def target_dictionary(self):
        return self.dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def return_date_prediction_list(self):
        date_prediction_list=[]
        for year_i in range(1987,2008):
            start_date=datetime(year_i,1,1)
            if year_i==2007:
                end_date=datetime(year_i,6,19)
            else:
                end_date=datetime(year_i,12,31)
            d=start_date
            dates=[start_date]
            while d < end_date:
                d += timedelta(days=1)
                dates.append(d)
            for date in dates:
                date_prediction_list.append(date.strftime("%Y-%m-%d"))
        return date_prediction_list

    def return_date2tempexp_dict(self, temp_granularity):
        date_prediction_list = self.date_prediction_list
        date2tempexp = dict()
        if temp_granularity=="Year":
            for date in date_prediction_list:
                date2tempexp[date] = date[:4]
        if temp_granularity=="Month":
            for date in date_prediction_list:
                month_string = date[:7]
                month_tempexp = format_date(iso8601.parse_date(month_string), "MMMM, yyyy", locale='en').strip()
                date2tempexp[date] = month_tempexp
        if temp_granularity=="Day":
            for idx,date in enumerate(date_prediction_list):
                day_tempexp = format_date(iso8601.parse_date(date), "MMMM dd, yyyy", locale='en').strip()
                date2tempexp[date] = day_tempexp
        return date2tempexp

    def return_date2tokenids_dict(self):
        date2tokenids_dict = dict()
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        basic_tokenizer = BasicTokenizer(do_lower_case=False)
        for date,tempexp in self.date2tempexp.items():
            tokens = basic_tokenizer.tokenize(tempexp)
            ids = torch.IntTensor(len(tokens))
            for i,token in enumerate(tokens):
                subtokens = tokenizer.tokenize(token)[0]
                idx = self.dictionary.index(subtokens)
                ids[i] = idx
            date2tokenids_dict[date] = ids
        return date2tokenids_dict

    def return_date2idx_dict(self, temp_granularity):
        date_prediction_list=self.date_prediction_list
        date_to_idx_dict=dict()
        if temp_granularity=="Year":
            for date in date_prediction_list:
                date_to_idx_dict[date]=int(date[:4])-1987
        if temp_granularity=="Month":
            for date in date_prediction_list:
                month_index=(int(date[:4])-1987)*12+int(date[5:7])-1
                date_to_idx_dict[date]=month_index
        if temp_granularity=="Day":
            for idx,date in enumerate(date_prediction_list):
                date_time=date
                date_to_idx_dict[date_time]=idx
        if temp_granularity=="Week":
            week_idx=0
            for date_infor in ['1987-01-01', '1987-01-02', '1987-01-03', '1987-01-04']:
                date_to_idx_dict[date_infor]=week_idx
            week_idx+=1
            date_format = "%Y-%m-%d"
            next_date='1987-01-04'
            new_week_last_date=(datetime.strptime('1987-01-04', date_format)+timedelta(days=7)).strftime("%Y-%m-%d")
            last_date="2007-06-19"
            while (next_date!=last_date):
                next_dateime=datetime.strptime(next_date, date_format)+timedelta(days=1)
                next_date=next_dateime.strftime("%Y-%m-%d")
                if next_date!=new_week_last_date:
                    date_to_idx_dict[next_date]=week_idx
                else:
                    new_week_last_date=(datetime.strptime(next_date, date_format)+timedelta(days=7)).strftime("%Y-%m-%d")
                    date_to_idx_dict[next_date]=week_idx
                    week_idx+=1
        return date_to_idx_dict

    def load_dataset(self, split):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        loaded_datasets = []
        path = os.path.join(self.args.data, split)
        #path='0_Corpus/1_Pretraining_Preprocessing/2_Preprocessed_Data/.../train|valid'

        if IndexedInMemoryDataset.exists(path):
            ds = IndexedInMemoryDataset(path, fix_lua_indexing=False)
            tokens = ds.buffer
            
        else:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        print(f"\n****************Use NSP:{not self.no_nsp}****************")
        block_cls = BlockPairDataset if not self.no_nsp else BlockDataset
        #self.no_nsp=False, block_cls=BlockPairDataset
        #check fairseq.data.temp_bert_dataset

        with data_utils.numpy_seed(self.seed):
            loaded_datasets.append(
                block_cls(
                    tokens, 
                    ds.sizes,
                    
                    ##Add these three arguments!
                    self.args.docid2timestamp_dir,
                    self.dictionary,
                    self.date2tokenids_dict,
                    self.date2idx_dict,
                    self.args.tokens_per_sample,        #tokens_per_sample=512
                    pad=self.dictionary.pad(),          #0
                    cls_idx=self.dictionary.cls(),      #101
                    mask=self.dictionary.mask(),        #103
                    sep=self.dictionary.sep(),          #102
                    break_mode='doc',                   #break_mode='doc'
                    short_seq_prob=self.short_seq_prob, #short_seq_prob=0.0
                    tag_map=None                        #tag_map = None
                ))

       
        print('| {} {} {} examples'.format(self.args.data, split, len(loaded_datasets[-1])))
        
        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets)
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets])

        dataset_cls = TempBertDataset if not self.no_nsp else NoNSPTempBertDataset
        #self.no_nsp=False -> dataset_cls = TempBertDataset
        #self.args.shuffle_instance=False

        self.datasets[split] = dataset_cls(dataset, sizes, self.dictionary, shuffle=False, seed=self.seed, args=self.args)
