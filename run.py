# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import

import copy
import os
import sys

import matplotlib.pyplot as plt

from metrics import bleu, rouge
import meteor
import pickle
import torch
import json
import networkx as nx
import random
import logging
import argparse
import numpy as np
from node2vec import Node2Vec
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq, Lstm2Lstm
from my_model import my_model
from transformer import my_transformer_model
from point_transformer import my_transformer_with_pointer_model
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
import math
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from torch.utils.data.distributed import DistributedSampler
from get_free_gpu import get_device

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
# get_device(memory_need=23000)
torch.set_num_threads(10)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class CosineAnnealing():
    def __init__(self, T, init_lr=0.5):
        super(CosineAnnealing, self).__init__()
        self.T = T
        self.init_lr = init_lr

    def get_temp(self, step):
        return 0.5 * (1 + math.cos(math.pi * step / self.T)) * self.init_lr

# class Example(object):
#     """A single training/test example."""
#     def __init__(self,
#                  idx,
#                  source,
#                  paths,
#                  target,
#                  ):
#         self.idx = idx
#         self.source = source
#         self.paths = paths
#         self.target = target
# def read_examples_xg(filename, sample_num = 23):
#     """Read examples from filename."""
#     examples = []
#     with open(filename + '.jsonl',encoding="utf-8") as f,\
#             open(filename + '_paths.jsonl') as p:
#         paths = json.load(p)
#         for idx, line in enumerate(f):
#             # if idx > sample_num:
#             #     break
#             line = line.strip()
#             js = json.loads(line)
#             if 'idx' not in js:
#                 js['idx']=idx
#             code = ' '.join(js['code_tokens']).replace('\n',' ')
#             code = ' '.join(code.strip().split())
#             path = paths[idx]
#             nl = ' '.join(js['docstring_tokens']).replace('\n','')
#             nl = ' '.join(nl.strip().split())
#             examples.append(
#                 Example(
#                         idx = idx,
#                         source = code,
#                         paths = path,
#                         target = nl,
#                         )
#             )
#     return examples
#
# def read_examples(filename, sample_num=47):
#     """Read examples from filename."""
#     examples = []
#     code_file = os.path.join(filename, 'code.txt')
#     nl_file = os.path.join(filename, 'nl.txt')
#     path_file = filename + '_paths.jsonl'
#     with open(code_file, encoding="utf-8") as f, \
#             open(nl_file, encoding="utf-8") as g, \
#             open(path_file) as p:
#         # if '/py/' in filename:
#         #     codes = json.load(f)
#         # if 'java' in filename:
#         codes = f.readlines()
#         summary = g.readlines()
#         paths = json.load(p)
#         for idx, code in enumerate(codes):
#             # if idx > sample_num:
#             #     break
#             code = code.strip()
#             path = paths[idx]
#             nl = summary[idx].strip()
#             examples.append(
#                 Example(
#                     idx=idx,
#                     source=code,
#                     paths=path,
#                     target=nl,
#                 )
#             )
#     return examples

# class InputFeatures(object):
#     """A single training/test features for a example."""
#     def __init__(self,
#                  example_id,
#                  source_ids,
#                  path_ids,
#                  target_ids,
#                  ):
#         self.example_id = example_id
#         self.source_ids = source_ids
#         self.path_ids = path_ids
#         self.target_ids = target_ids
# def convert_examples_to_features(examples, tokenizer, args, stage=None):
#     """convert examples to token ids"""
#     features = []
#     for example_index, example in enumerate(examples):
#         # source_tokens = tokenizer.tokenize("1 " + example.source)[1:args.max_source_length-1]
#         # source_tokens = [tokenizer.cls_token,"<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
#         # source_tokens = example.source.split()[:args.max_source_length-2]
#         source_tokens = []
#         all_sentences_ids = []
#         all_sentences_ids.append(args.max_path_length * [19])
#         for path in example.paths:
#             sentences_tokens = path.split('<split>')
#             first_tokenize = tokenizer.tokenize(sentences_tokens[0])
#             source_tokens += first_tokenize
#             sentences_tokens = tokenizer.tokenize(' '.join(sentences_tokens))[: args.max_path_length - 1] + [tokenizer.sep_token]
#
#             sentences_ids = tokenizer.convert_tokens_to_ids(sentences_tokens)
#             padding_length = args.max_path_length - len(sentences_ids)
#             sentences_ids += [tokenizer.pad_token_id] * padding_length
#             for _ in range(len(first_tokenize)):
#                 all_sentences_ids.append(sentences_ids)
#         source_tokens = ["<mask0>"] + source_tokens[:args.max_source_length - 2] + [tokenizer.sep_token]
#         source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
#         padding_length = args.max_source_length - len(source_ids)
#         source_ids += [tokenizer.pad_token_id] * padding_length
#
#         all_sentences_ids = all_sentences_ids[:args.max_source_length - 1]
#         padding_length = args.max_source_length - len(all_sentences_ids)
#         pad_seq = [[19] + [tokenizer.sep_token_id] + (args.max_path_length - 2) * [tokenizer.pad_token_id]]
#         all_sentences_ids += padding_length * pad_seq
#
#         # target
#         if stage == "test":
#             target_tokens = tokenizer.tokenize("None")
#         else:
#             target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
#         target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
#         target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
#         padding_length = args.max_target_length - len(target_ids)
#         target_ids += [tokenizer.pad_token_id] * padding_length
#
#         if example_index < 5:
#             if stage == 'train':
#                 logger.info("*** Example ***")
#                 logger.info("idx: {}".format(example.idx))
#
#                 logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
#                 logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
#
#                 # logger.info("path_tokens: {}".format([x.replace('\u0120', '_') for x in sentences_tokens]))
#                 # logger.info("path_ids: {}".format(' '.join(map(str, sentences_ids))))
#
#                 logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
#                 logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
#
#         features.append(
#             InputFeatures(
#                 example_index,
#                 source_ids,
#                 all_sentences_ids,
#                 target_ids,
#             )
#         )
#     return features

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 target
                 ):
        self.idx = idx
        self.target = target

def read_examples(filename, sample_num=47):
    """Read examples from filename."""
    examples = []
    filename = filename + '.jsonl'
    with open(filename, encoding="utf-8") as g:
        code_and_summarys = json.load(g)
        # summary = g.readlines()
        for idx, code_and_summary in enumerate(code_and_summarys):
            nl = code_and_summary['summary']
            examples.append(
                Example(
                    idx=idx,
                    target=nl
                )
            )
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 target_ids
                 ):
        self.example_id = example_id
        self.target_ids = target_ids

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    """convert examples to token ids"""
    features = []
    for example_index, example in enumerate(examples):
        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))

        features.append(
            InputFeatures(
                example_index,
                target_ids,
            )
        )
    return features

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--reload", action='store_true',
                        help="Whether to load the model.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_path_length", default=32, type=int,
                        help="The maximum total path sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=40, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_node_length", default=256, type=int,
                        help="The maximum node numbers. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_layers", default=6, type=int,
                        help="The number of layers of Transformer encoder and decoder.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gamma", default=0.5, type=float,
                        help="gamma.")
    # parser.add_argument("--num_cl_train_epochs", default=30, type=int,
    #                     help="Total number of contrastive learning training epochs to perform.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--dynamic_coefficient_step', type=int, default=5,
                        help='the length of history windows on coefficient step')

    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


    # Set seed
    set_seed(args.seed)

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # tensorboard
    writer = SummaryWriter(args.output_dir)

    # train_examples = read_examples(args.train_filename)

    logger.info("Training/evaluation parameters %s", args)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # build model
    # with open('/data/yzzhao/pythonCode/Code_Intelligence/data/python/summary_tokenizer.pkl', 'rb') as f:
    #     tgt_tokenizer = pickle.load(f)
    # tokenizer = Tokenizer.from_file('/data/yzzhao/pythonCode/Code_Intelligence/data/python/summary_tokenizer.pkl')
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    with open(os.path.join('/'.join(args.train_filename.split('/')[:-1]), 'special_tokens.jsonl')) as sp:
        tokenizer.add_special_tokens(json.load(sp))
    config = RobertaConfig.from_pretrained(args.model_name_or_path)

    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    # encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    #
    # model = Seq2Seq(encoder=encoder, decoder=encoder, config=config,
    #                 beam_size=args.beam_size, max_length=args.max_target_length,
    #                 sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0], eos_id=tokenizer.sep_token_id)

    # model = Lstm2Lstm(tokenizer.vocab_size,
    #                   emb_size=768,
    #                   hidden_size=512,
    #                   beam_size=10,
    #                   max_length=args.max_target_length,
    #                   sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
    #                   eos_id=tokenizer.sep_token_id)

    model = my_model(len(tokenizer.get_vocab()), max_source_len=args.max_source_length, d_model=512, hidden_size=512, nhead=8, num_layers=args.num_layers, gamma=args.gamma, beam_size=10,
                     max_length=args.max_target_length, sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
                     eos_id=tokenizer.sep_token_id)
    # model = my_transformer_model(tokenizer.vocab_size, emb_size=768, hidden_size=768, nhead=8, beam_size=10, max_length=args.max_target_length, sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0], eos_id=tokenizer.sep_token_id)
    model.to(args.device)
    r = rouge.Rouge()
    m = meteor.Meteor()
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_path_ids = np.load(args.train_filename + '_paths_wostop.npy').astype(int)
        # all_path_ids = torch.tensor([f.path_ids for f in train_features], dtype=torch.long)
        all_path_ids = torch.tensor(all_path_ids, dtype=torch.long)
        # all_source_ids = torch.tensor(copy.copy(all_path_ids[:,:,0]), dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        # train_data = TensorDataset(all_source_ids, all_path_ids, all_target_ids)
        train_data = TensorDataset(all_path_ids, all_target_ids)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size // args.gradient_accumulation_steps,
                                      shuffle=True, pin_memory=True, num_workers=8)
        # train_sampler = RandomSampler(train_data)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=True)
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=int(
        #                                                 len(train_dataloader) * args.num_train_epochs * 0.1),
        #                                             num_training_steps=len(train_dataloader) * args.num_train_epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=len(train_dataloader) * args.num_train_epochs)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x:0.95**x)
        CosA = CosineAnnealing(args.num_train_epochs)
        patience, best_bleu, losses, acces, dev_dataset = 0, 0, [], [], {}

        # reload model
        start_epoch = 0
        if args.reload and os.path.exists(args.output_dir):
            checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
            model_data = torch.load(output_dir)
            model_to_load = model.module if hasattr(model, 'module') else model
            # model_to_load.load_state_dict(model_data)
            model_to_load.load_state_dict(model_data['model_dict'])
            optimizer.load_state_dict(model_data['optimizer'])
            scheduler.load_state_dict(model_data['scheduler'])
            best_bleu = model_data['best_bleu']
            start_epoch = model_data['epoch']
            logger.info("  Reload model from: %s", args.output_dir)

        # Start training
        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        model.train()
        # for epoch in range(args.num_train_epochs):
        #     for idx,batch in enumerate(train_dataloader):
        #         batch = tuple(t.to(device) for t in batch)
        #         source_ids, path_ids, target_ids = batch

        # step1:contrastive learning
        #         loss, acc = model(source_ids=source_ids, path_ids=path_ids, target_ids=target_ids, temperature=CosA.get_temp(epoch), tasks='clt')
        #         if args.n_gpu > 1:
        #             loss = loss.mean()  # mean() to average on multi-gpu.
        #             acc = acc.mean()
        #         if args.gradient_accumulation_steps > 1:
        #             loss = loss / args.gradient_accumulation_steps
        #             acc = acc / args.gradient_accumulation_steps

        #         losses.append(loss.item())
        #         acces.append(acc.item())
        #         loss.backward()
        #         if len(losses) % args.gradient_accumulation_steps == 0:
        #             # Update parameters
        #             optimizer.step()
        #             optimizer.zero_grad()
        #             scheduler.step()
        #             if len(losses) // args.gradient_accumulation_steps % 10 == 0:
        #                 logger.info("task: clt epoch {} step {} loss {} acc {}".format(epoch,
        #                                              len(losses)//args.gradient_accumulation_steps,
        #                                              round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4),
        #                                              round(np.mean(acces[-100*args.gradient_accumulation_steps:]),4)))
        # output_dir = os.path.join(args.output_dir, 'trained_contrastive_learning')
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # output_model_file = os.path.join(output_dir, "cl_pytorch_model.bin")
        # torch.save(model_to_save.state_dict(), output_model_file)
        # checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
        # output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        # model_to_load = model.module if hasattr(model, 'module') else model
        # model_to_load.load_state_dict(torch.load(output_dir))

        # all_pos, all_neg = np.zeros((1)), np.zeros((1))
        # model.eval()
        # for idx, batch in enumerate(train_dataloader):
        #     batch = tuple(t.to(device) for t in batch)
        #     path_ids, target_ids = batch
        #     pos, neg = model.cal_dis(path_ids=path_ids, target_ids=target_ids,
        #                              temperature=0.33, tasks='gst')
        #     all_pos = np.concatenate([all_pos, pos], axis=0)
        #     all_neg = np.concatenate([all_neg, neg], axis=0)
        # all_pos, all_neg = all_pos[1:], all_neg[1:]
        # # np.save('/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/woAB_4_Bina_contrast_avgEmb_trans_copy/base_pos_dis.npy', all_pos)
        # # np.save('/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/woAB_4_Bina_contrast_avgEmb_trans_copy/base_neg_dis.npy', all_neg)
        # # plt.title('Difference in distance between positive and negative pairs')
        # label = 'Positive pairs', 'Positive pairs(base)', 'Negative pairs', 'Negative pairs(base)'
        # # label = 'Positive pairs base', 'Negative pairs', 'Negative pairs base'
        # base_all_pos = np.load('/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/woAB_4_Bina_contrast_avgEmb_trans_copy/base_pos_dis.npy')
        # base_all_neg = np.load(
        #     '/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/woAB_4_Bina_contrast_avgEmb_trans_copy/base_neg_dis.npy')
        # print(all_neg.mean())
        # print(all_pos.mean())
        # print(base_all_neg.mean())
        # print(base_all_pos.mean())
        # plt.boxplot([all_pos, base_all_pos, all_neg, base_all_neg], showfliers=False, labels=label, showmeans=True)
        # # plt.boxplot([base_all_pos, all_neg, base_all_neg], showfliers=False, labels=label)
        # # plt.show()
        # plt.savefig('/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/dis_base.png')

        for epoch in range(start_epoch, args.num_train_epochs):
            # for name, param in model.named_parameters():
            #     writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=epoch)
            #     writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch)

            for idx, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                path_ids, target_ids = batch
                # step2:generate tasks
                # model.eval()

                # cop_loss,pre_loss,clt_loss,contrast_acc, _, _ = model(path_ids=path_ids, target_ids=target_ids,
                #                    temperature=max(CosA.get_temp(epoch), 0.33), tasks='gst')
                # loss = cop_loss + pre_loss + clt_loss

                # cop_loss, pre_loss, _, _ = model(path_ids=path_ids, target_ids=target_ids,
                #                                                          temperature=max(CosA.get_temp(epoch), 0.33),
                #                                                          tasks='gst')
                # loss = cop_loss + pre_loss

                loss, _, _ = model(path_ids=path_ids, target_ids=target_ids, temperature=max(CosA.get_temp(epoch), 0.33), tasks='gst')
                step = len(losses)


                # if epoch < 20:
                #     loss = cop_loss + pre_loss + clt_loss
                # elif epoch < 85:
                #     loss = cop_loss + pre_loss
                # else:
                #     loss = pre_loss

                # loss, _, _ = model(source_ids=source_ids, path_ids=path_ids, target_ids=target_ids,
                #                    temperature=0.2, tasks='gst')
                # loss, _, _ = model(source_ids=source_ids, target_ids=target_ids)
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    # cop_loss = cop_loss.mean()
                    # pre_loss = pre_loss.mean()
                    # clt_loss = clt_loss.mean()
                    # contrast_acc = contrast_acc.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # writer.add_scalar('Loss/train_cop_loss', cop_loss, step)
                # writer.add_scalar('Loss/train_loss', pre_loss, step)
                # writer.add_scalar('Loss/train_clt_loss', clt_loss, step)
                # writer.add_scalar('Loss/train_contrast_acc', contrast_acc, step)
                # writer.add_scalar('Parameter/train_alpha', model.alpha, step)

                losses.append(loss.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    # scheduler.step()
                    if len(losses) // args.gradient_accumulation_steps % 10 == 0:
                        logger.info("task: gst epoch {} step {} loss {}".format(epoch,
                                                                                len(losses) // args.gradient_accumulation_steps,
                                                                                round(np.mean(losses[-100 * args.gradient_accumulation_steps:]), 4)))
            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    all_path_ids = np.load(args.dev_filename + '_paths_wostop.npy').astype(int)
                    # all_path_ids = torch.tensor([f.path_ids for f in train_features], dtype=torch.long)
                    all_path_ids = torch.tensor(all_path_ids, dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_path_ids, all_target_ids)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                # eval_sampler = SequentialSampler(eval_data)
                # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, pin_memory=True, num_workers=8)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                all_cop_loss, all_pre_loss, all_clt_loss, all_contrast_acc = 0, 0, 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    path_ids, target_ids = batch

                    with torch.no_grad():
                        # cop_loss,pre_loss,clt_loss,contrast_acc, loss, num = model(path_ids=path_ids, target_ids=target_ids,
                        #                      temperature=max(CosA.get_temp(epoch), 0.33), tasks='gst')
                        pre_loss, loss, num = model(path_ids=path_ids, target_ids=target_ids,
                                                    temperature=max(CosA.get_temp(epoch), 0.33), tasks='gst')
                        # _, loss, num = model(source_ids=path_ids, target_ids=target_ids)

                    # all_cop_loss += cop_loss.sum().item()
                    # all_pre_loss += pre_loss.sum().item()
                    # all_clt_loss += clt_loss.sum().item()
                    # all_contrast_acc += contrast_acc.sum().item()

                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                model.train()

                # writer.add_scalar('Loss/eval_cop_loss', all_cop_loss, epoch)
                # writer.add_scalar('Loss/eval_loss', all_pre_loss, epoch)
                # writer.add_scalar('Loss/eval_clt_loss', all_clt_loss, epoch)
                # writer.add_scalar('Loss/eval_contrast_acc', all_contrast_acc, epoch)
                # writer.add_scalar('Parameter/eval_alpha', model.alpha, epoch)

                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    # sample_idx = np.random.randint(0, len(eval_examples), min(1000, len(eval_examples)))
                    sample_idx = random.sample(range(len(eval_examples)), min(1000, len(eval_examples)))
                    eval_examples = [eval_examples[idx] for idx in sample_idx]
                    # eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                    # eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    all_path_ids = np.load(args.dev_filename + '_paths_wostop.npy').astype(int)[sample_idx]
                    # all_path_ids = torch.tensor([f.path_ids for f in train_features], dtype=torch.long)
                    all_path_ids = torch.tensor(all_path_ids, dtype=torch.long)
                    # all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    # all_path_ids = torch.tensor([f.path_ids for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_path_ids)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data


                # eval_sampler = RandomSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, pin_memory=True, num_workers=8)
                # eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, pin_memory=True, num_workers=4)

                model.eval()
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    path_ids = batch[0]
                    with torch.no_grad():
                        preds = model(path_ids)
                        # preds = model(source_ids)
                        # convert ids to text
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions = []
                with open(args.output_dir + "/dev.output", 'w') as f, open(args.output_dir + "/dev.gold", 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx) + '\t' + ref)
                        f.write(str(gold.idx) + '\t' + ref + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target + '\n')

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
                dev_rouge = round(r.compute_score(predictionMap, goldMap)[0], 4)
                dev_meteor = round(m.compute_score(predictionMap, goldMap)[0], 4)
                dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s ;  %s = %s ;  %s = %s " % (
                "bleu-4", str(dev_bleu), "rouge-l", str(dev_rouge), "meteor", str(dev_meteor)))
                # logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  " + "*" * 20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save({'epoch':epoch,
                                'best_bleu':best_bleu,
                                'optimizer':optimizer.state_dict(),
                                'scheduler':scheduler.state_dict(),
                                'model_dict':model_to_save.state_dict()}, output_model_file)
                    patience = 0
                else:
                    scheduler.step()
                    patience += 1
                    if patience == 50:
                        break

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_data = torch.load(output_dir)
        model_to_load.load_state_dict(model_data['model_dict'])
        # model_to_load.load_state_dict(torch.load(output_dir))

        eval_examples = read_examples(args.test_filename)
        # eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
        all_path_ids = np.load(args.test_filename + '_paths_wostop.npy').astype(int)
        all_path_ids = torch.tensor(all_path_ids, dtype=torch.long)
        # all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
        # all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        # all_path_ids = torch.tensor([f.path_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_path_ids)

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, pin_memory=True, num_workers=8)

        model.eval()
        p = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            path_ids = batch[0]
            with torch.no_grad():
                preds = model(path_ids=path_ids, tasks='gst')
                # convert ids to text
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)

        model.train()
        predictions = []
        with open(args.output_dir + "/test.output", 'w') as f, open(args.output_dir + "/test.gold", 'w') as f1:
            for ref, gold in zip(p, eval_examples):
                predictions.append(str(gold.idx) + '\t' + ref)
                f.write(str(gold.idx) + '\t' + ref + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target + '\n')

        (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test.gold"))
        test_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        test_rouge = round(r.compute_score(predictionMap, goldMap)[0], 4)
        test_meteor = round(m.compute_score(predictionMap, goldMap)[0], 4)
        # dev_meteor=round(m.compute_score(predictionMap, goldMap)[0],2)
        logger.info("  %s = %s ;  %s = %s ;  %s = %s " % (
        "bleu-4", str(test_bleu), "rouge-l", str(test_rouge), "meteor", str(test_meteor)))
        # logger.info("  %s = %s ;  %s = %s" % ("bleu-4", str(dev_bleu), "rouge-l", str(dev_rouge)))
        logger.info("  " + "*" * 20)


if __name__ == "__main__":
    main()


