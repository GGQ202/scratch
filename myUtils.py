import copy
import re

import torch
from torch.utils.data import TensorDataset

from processor import InputFeatures


def process_article(article, args, tokenizer, max_seq_length,
                    sep_token='[SEP]', cls_token='[CLS]', sequence_a_segment_id=0, pad_token=0):
    sentences = re.split(r"[。？！ ；!?]", article)
    features = []
    tokens_list = []
    for sentence in sentences:
        if sentence == "":
            continue
        tokens = tokenizer.tokenize(sentence)
        tokens_list.append(copy.deepcopy(tokens))
        tokens += [sep_token]
        tokens = [cls_token] + tokens
        segment_ids = [sequence_a_segment_id] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_len = len(input_ids)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        # padding
        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                      segment_ids=segment_ids, label_ids=None))
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_label_ids = torch.ones_like(all_input_ids)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset, tokens_list
