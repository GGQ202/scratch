import platform

import torch
from transformers import WEIGHTS_NAME, BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizer
from models.bert_for_ner import BertCrfForNer


class Args:
    def __init__(self):
        self.platform = platform.system().lower()
        self.no_cuda = True
        self.device = None
        self.n_gpu = torch.cuda.device_count()
        self.label_list = ["X", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                           'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                           'O', 'S-NAME', 'S-ORG', 'S-RACE', "[START]", "[END]"]
        self.config_class = BertConfig
        self.model_class = BertCrfForNer
        self.tokenizer_class = BertTokenizer
        self.task = "do_train"
        self.train_batch_size = 8
        self.num_train_epochs = None
        self.model_name_or_path = 'prev_trained_model/bert-base-chinese'
        self.data_dir = "./datasets/cner"
        self.train_max_seq_length = 128
        self.model_type = 'bert'
        self.max_steps = -1
        self.gradient_accumulation_steps = 1
        self.num_train_epochs = 10
        self.weight_decay = 0.01
        self.learning_rate = 5e-5
        self.crf_learning_rate = 5e-5
        self.warmup_proportion = 0.1
        self.warmup_steps = None
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.output_dir = "output"
