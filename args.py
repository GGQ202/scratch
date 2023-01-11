import platform

import torch
from transformers import WEIGHTS_NAME, BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizer
from models.bert_for_ner import BertCrfForNer


class Args:
    def __init__(self):
        self.platform = platform.system().lower()
        self.no_cuda = True if self.platform=='windows' else False
        self.device = None
        self.n_gpu = torch.cuda.device_count()
        label = [["X", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                'O', 'S-NAME', 'S-ORG', 'S-RACE', "[START]", "[END]"],
                ['B_time', 'I_time', 'B_place', 'I_place', 'B_person', 'I_person',
                'B_num', 'I_num', 'B_culture', 'I_culture', 'B_symbol', 'I_symbol',
                'B_direct', 'I_direct', 'B_func', 'I_func', 'B_act', 'I_act',
                'B_obj', 'I_obj', 'O', '[START]', '[END]'],
                ['I_disease', 'B_symptom', 'B_drug', 'I_crowd', 'B_crowd',
                  'B_feature', 'I_department', 'B_department', 'I_drug',
                 'B_test', 'B_treatment', 'I_time', 'B_time', 'I_feature',
                 'B_body', 'O', 'I_body', 'I_physiology', 'B_physiology',
                 'I_test', 'B_disease', 'I_treatment', 'I_symptom', '[START]', '[END]']]
        self.label_list = label[2]

        self.config_class = BertConfig
        self.model_class = BertCrfForNer
        self.tokenizer_class = BertTokenizer
        self.do_train = True
        self.do_eval = False
        self.do_predict = False
        self.do_console_predict = True
        self.batch_size = 8
        self.num_train_epochs = None
        self.model_name_or_path = 'prev_trained_model/bert-base-chinese'
        self.data_dir = "./datasets/mydata"
        self.output_dir = "output"
        self.train_max_seq_length = 128
        self.eval_max_seq_length = 512
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

        self.id2label = None
        self.label2id = None
