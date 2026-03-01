import math
from tqdm import tqdm
import numpy as np
import torch
from math import sqrt, log
from . import data_utils, FairseqDataset
from itertools import chain
import json
import random
import pickle
from collections import defaultdict
from fairseq.data.masking import ParagraphInfo, BertRandomMaskingScheme
random.seed(0)

class BlockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokens,
        sizes,
        docid2timestamp_dir,
        dictionary,
        date2tokenids_dict,
        date2idx_dict,

        block_size,          #tokens_per_sample=512
        pad,                 #0
        cls_idx,             #101
        mask,                #103
        sep,                 #102
        break_mode="doc",    #break_mode='doc'
        short_seq_prob=0.1,  #short_seq_prob=0.0
        tag_map=None         #tag_map = None
    ):
        super().__init__()
        self.tokens = tokens
        self.total_size = len(tokens)
        self.pad = pad
        self.cls = cls_idx
        self.mask = mask
        self.sep = sep
        self.break_mode = break_mode
        self.tag_map = tag_map

        docid2timestamp = pickle.load(open(docid2timestamp_dir, 'rb'))

        self.block_indices = []
        self.sents = []
        self.sizes = []
        self.timestamp_ids = []
        self.timestamp_labels = []
        self.sentidx2pos_dict = dict()
        self.docids = []

        if break_mode == "sentence":
            curr = 0
            for sz in sizes:
                if sz == 0:
                    continue
                self.block_indices.append((curr, curr + sz))
                curr += sz
            for curr in range(len(self.block_indices)):
                sent = self.block_indices[curr]
                if sent[1] - sent[0] <= max_num_tokens:
                    self.sents.append(sent)
                    self.sizes.append(sent[1] - sent[0] + 2)
        elif break_mode == "doc":
            assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
            curr = 0
            cur_doc = []
            for sz in sizes:
                if sz == 0:
                    if len(cur_doc) == 0: continue
                    self.block_indices.append(cur_doc)
                    cur_doc = []
                else:
                    cur_doc.append((curr, curr + sz))
                curr += sz
            
            max_num_tokens = block_size - 2 # Account for [CLS], [SEP]

            for doc_idx, doc in enumerate(self.block_indices):
                current_chunk = []
                curr = 0
                start_idx, end_idx = doc[-1]
                docid = ""
                for r in self.tokens[start_idx:end_idx]:
                    docid+=dictionary.symbols[r].replace("##","")
                timestamp = docid2timestamp[docid]
                timestamp = timestamp[:4]+"-"+timestamp[4:6]+"-"+timestamp[6:8]
                ts_ids = date2tokenids_dict[timestamp]
                ts_labels = date2idx_dict[timestamp]
                
                doc = doc[:-1]
                sentidx2pos = dict()
                for sent_i, sent in enumerate(doc):
                    sentidx2pos[sent_i] = sent
                self.sentidx2pos_dict[docid] = sentidx2pos

                while curr < len(doc):
                    sent = doc[curr]
                    if sent[1] - sent[0] <= max_num_tokens:
                        current_chunk.append(sent)
                        current_length = current_chunk[-1][1] - current_chunk[0][0]
                        if curr == len(doc) - 1 or current_length > max_num_tokens:
                            if current_length > max_num_tokens:
                                current_chunk = current_chunk[:-1]
                                curr -= 1
                            if len(current_chunk) > 0:
                                sent = (current_chunk[0][0], current_chunk[-1][1])
                                self.docids.append(docid)
                                self.sents.append(sent)
                                self.sizes.append(sent[1] - sent[0] + 2)
                                self.timestamp_ids.append(ts_ids)
                                self.timestamp_labels.append(ts_labels)
                            current_chunk = []
                    curr += 1 
        else:
            raise ValueError("break_mode = %s not supported." % self.break_mode)

    def __getitem__(self, index):
        return self.sents[index]

    def __len__(self):
        return len(self.sizes)

class NoNSPTempBertDataset(FairseqDataset):
    """
    A wrapper around BlockDataset for BERT data.
    Args:
        dataset (BlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """
    def __init__(self, dataset, sizes, vocab, shuffle, seed, args=None):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.masking_schemes = []
        self.paragraph_info = ParagraphInfo(vocab)
        self.args = args
        scheme = BertRandomMaskingScheme(args, self.dataset.tokens, self.dataset.pad, self.dataset.mask)
        self.masking_schemes.append(scheme)

        self.temp_masked_ratio = self.args.temp_masked_ratio
        self.docid2sentidx2temptokenizeinfor_dict = pickle.load(open(self.args.docid2sentidx2temptokenizeinfor_dict_file,'rb'))
        
    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            block = self.dataset[index]
        docid = self.dataset.docids[index]
        
        tokens_ids = self.dataset.tokens[block[0]:block[1]].copy()
        sentidx2temptokenizeinfor_dict = self.docid2sentidx2temptokenizeinfor_dict[docid]
        sentidx2pos_dict = self.dataset.sentidx2pos_dict[docid]
        temp_infor_list = self.return_temp_infor_list(tokens_ids, block, sentidx2temptokenizeinfor_dict, sentidx2pos_dict)

        tagmap = self.dataset.tag_map[block[0]:block[1]] if self.dataset.tag_map is not None else None
        masked_block, masked_tgt, pair_targets = self._mask_block(tokens_ids, temp_infor_list, self.temp_masked_ratio, tagmap)
        #pair_targets are all None

        input_ids = np.concatenate([[self.vocab.cls()],masked_block,[self.vocab.sep()],])
        input_mask = np.ones_like(input_ids)
        segment_ids = np.zeros(block[1] - block[0] + 2)
        lm_label_ids = np.concatenate([[self.vocab.pad()], masked_tgt, [self.vocab.pad()]])

        input_ids = np.pad(input_ids, [0,self.args.tokens_per_sample-len(input_ids)], 'constant')
        input_mask = np.pad(input_mask, [0,self.args.tokens_per_sample-len(input_mask)], 'constant')
        segment_ids = np.pad(segment_ids, [0,self.args.tokens_per_sample-len(segment_ids)], 'constant')
        lm_label_ids = np.pad(lm_label_ids, [0,self.args.tokens_per_sample-len(lm_label_ids)], 'constant')

        #Add timestamp information
        ts_ids = self.dataset.timestamp_ids[index]
        ts_labels = self.dataset.timestamp_labels[index]

        results = (torch.tensor(input_ids, dtype=torch.int64),
                   torch.tensor(input_mask, dtype=torch.int64),
                   torch.tensor(segment_ids, dtype=torch.int64),
                   torch.tensor(lm_label_ids, dtype=torch.int64),
                   torch.tensor(ts_ids, dtype=torch.int64),
                   torch.tensor(ts_labels, dtype=torch.int64),
                   torch.tensor(index, dtype=torch.int64))
        return results

    def return_temp_infor_list(self, tokens_ids, block, sentidx2temptokenizeinfor_dict, sentidx2pos_dict):
        temp_infor_list = []
        for sentidx,temptokenizeinfor in sentidx2temptokenizeinfor_dict.items():
            b_sent_pos, e_sent_pos = sentidx2pos_dict[sentidx]
            b_sent_pos = b_sent_pos-block[0]
            e_sent_pos = e_sent_pos-block[0]
            sent_tokens_ids = tokens_ids[b_sent_pos:e_sent_pos]
            if 0<=b_sent_pos<=e_sent_pos<=block[1]-block[0]:
                for tokenizeinfor in temptokenizeinfor:
                    sent_tokens_ids_str = "_".join(map(str, sent_tokens_ids))
                    temptokenize_ids_str = "_".join(map(str, tokenizeinfor[1]))
                    b_temp_pos = b_sent_pos+sent_tokens_ids_str[:sent_tokens_ids_str.index(temptokenize_ids_str)].count("_")
                    e_temp_pos = b_temp_pos+len(tokenizeinfor[1])
                    temp_infor_list.append([b_sent_pos, e_sent_pos, b_temp_pos, e_temp_pos, tokenizeinfor[1], tokenizeinfor[0]])
        return temp_infor_list

    def __len__(self):
        return len(self.dataset)

    def _collate(self, samples, pad_idx):
        if len(samples) == 0:
            return {}

        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )
        def merge_2d(key):
            return data_utils.collate_2d(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )
        pair_targets = merge_2d('pair_targets')

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'ntokens': sum(len(s['source']) for s in samples),
            'net_input': {
                'src_tokens': merge('source'),
                'segment_labels': merge('segment_labels'),
                'pairs': pair_targets[:, :, :2]
            },
            'lm_target': merge('lm_target'),
            'nsentences': samples[0]['source'].size(0),
            'pair_targets': pair_targets[:, :, 2:]
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return self._collate(samples, self.vocab.pad())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=12):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        source = self.vocab.dummy_sentence(tgt_len)
        segment_labels = torch.zeros(tgt_len, dtype=torch.long)
        pair_targets = torch.zeros((1, self.args.max_pair_targets + 2), dtype=torch.long)
        lm_target = source
        bsz = num_tokens // tgt_len

        return self.collater([
            {
                'id': i,
                'source': source,
                'segment_labels': segment_labels,
                'lm_target': lm_target,
                'pair_targets': pair_targets
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            return np.random.permutation(len(self))
        order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def _mask_block(self, sentence, temp_infor_list, temp_masked_ratio, tagmap):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        masking_scheme = random.choice(self.masking_schemes)
        block = masking_scheme.mask(sentence, temp_infor_list, temp_masked_ratio, tagmap)
        return block
