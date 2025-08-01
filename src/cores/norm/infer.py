#!/usr/bin/env python
# coding: utf-8
import torch
from data_handling import DataCollatorForNormSeq2Seq
from model_handling import EncoderDecoderSpokenNorm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Init tokenizer and model
from importlib.machinery import SourceFileLoader
import os

cache_dir='./invert_norm_ckpt'
tokenizer = (
    SourceFileLoader(
        "envibert.tokenizer", 
        os.path.join(cache_dir,'envibert_tokenizer.py')
    ).load_module().RobertaTokenizer(cache_dir)
)
model = EncoderDecoderSpokenNorm.from_pretrained(cache_dir).eval()
data_collator = DataCollatorForNormSeq2Seq(tokenizer)

# Infer sample
bias_list = ['scotland', 'covid', 'delta', 'beta']
input_list = [
    'ngày hai tám tháng tư cô vít bùng phát ở sờ cốt lờn chiếm tám mươi phần trăm là biến chủng đen ta và bê ta',
    'tôi muốn vay hai lăm triệu đồng'
]

# inputs = tokenizer(input_list)
# input_ids = inputs['input_ids']
# attention_mask = inputs['attention_mask']
if len(bias_list) > 0:
    bias = data_collator.encode_list_string(bias_list)
    bias_input_ids = bias['input_ids']
    bias_attention_mask = bias['attention_mask']
else:
    bias_input_ids = None
    bias_attention_mask = None

inputs = data_collator.encode_list_string(input_list)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "bias_input_ids": bias_input_ids,
    "bias_attention_mask": bias_attention_mask,
}

# Format input text **with** bias phrases

outputs = model.generate(**inputs, output_attentions=True, num_beams=1, num_return_sequences=1)

for output in outputs.cpu().detach().numpy().tolist():
    # print('\n', tokenizer.decode(output, skip_special_tokens=True).split(), '\n')
    print(tokenizer.sp_model.DecodePieces(tokenizer.decode(output, skip_special_tokens=True).split()))
# output: 28/4 covid bùng phát ở scotland chiếm 80 % là biến chủng delta và beta
