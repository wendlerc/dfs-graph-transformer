#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:02:39 2022

@author: chrisw
"""


from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
import torch
from transformers import BertConfig
from transformers import BertTokenizerFast
from transformers import BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import os


import argparse


parser = argparse.ArgumentParser()
parser.add_argument('train_file', type=str)
parser.add_argument('model_dir', type=str)

args = parser.parse_args()
        

paths = [args.train_file]

# Initialize a tokenizer
#tokenizer = ByteLevelBPETokenizer()
tokenizer = BertWordPieceTokenizer(
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)

tokenizer.train(files=paths, vocab_size=52000, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

os.makedirs(args.model_dir, exist_ok=True)
tokenizer.save_model(args.model_dir)

class SmilesDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = BertWordPieceTokenizer(args.model_dir+'/vocab.txt')
        tokenizer._tokenizer.post_processor = BertProcessing(("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[CLS]", tokenizer.token_to_id("[CLS]")))
        tokenizer.enable_truncation(max_length=512)

        self.examples = []
        
        
        src_files = [Path(paths[0])]
        for src_file in src_files:
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])
    
config = BertConfig(vocab_size=52000, 
                    hidden_size=600,
                    num_hidden_layers=6,
                    num_attention_heads=12,
                    intermediate_size=4*600,
                    max_position_embeddings=514,
                    type_vocab_size=1)
tokenizer = BertTokenizerFast.from_pretrained(args.model_dir, max_len=512)
model = BertForMaskedLM(config=config)
dataset = SmilesDataset()
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
training_args = TrainingArguments(
    output_dir=args.model_dir,
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_gpu_train_batch_size=64,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(args.model_dir)