import torch

from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import AST
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    changeidentifier,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser
import networkx as nx
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
from node2vec import Node2Vec
from tqdm import tqdm
import javalang
from transformers import RobertaTokenizer
import argparse
import random

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizer

def get_vocab(corpus_path):
    src_corpus = [os.path.join(corpus_path, type, 'split_code.txt') for type in ['train', 'dev', 'test']]
    # tgt_corpus = [os.path.join(corpus_path, type, 'nl.txt') for type in ['train', 'dev', 'test']]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    trainer = BpeTrainer(vocab_size=50000)

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(trainer=trainer, files=src_corpus)

    tokenizer.save(os.path.join(corpus_path, 'src_vacb.json'))

def clean_code(code):
    return code.replace(
        ' DCNL DCSP ', '\n\t'
    ).replace(
        ' DCNL  DCSP ', '\n\t'
    ).replace(
        ' DCNL   DCSP ', '\n\t'
    ).replace(
        ' DCNL ', '\n'
    ).replace(' DCSP ', '\t').replace('\t', '    ')

def split_by_whitespace(s):
    return s.split(" ")

def python_tokenize(line):
    tokens = re.split('\.|\(|\)|\:| |;|,|!|=|[|]', line)
    return [t for t in tokens if t.strip()]

def tokenize_source_code(code_string):
    """
    Generate a list of string after javalang tokenization.
    :param code_string: a string of source code
    :return:
    """
    code_string.replace("#", "//")
    try:
        tokens = list(javalang.tokenizer.tokenize(code_string))
        return [token.value for token in tokens]

    except:
        # logger.info(code_string)
        # logger.info(10 * "*")
        # with open("error.log", "a") as f:
        #     f.write(code_string)
        #     f.write("\n")
        #     f.write(traceback.format_exc())
        #     f.write("\n")
        #     f.write(20 * "*")
        #     f.write("\n")
        return None

def lower_case_str_arr(str_arr):
    return [tok.lower() for tok in str_arr]

def code_tokens_replace_str_num(sequence):
    tokens = []
    for s in sequence:
        if s[0] == '"' and s[-1] == '"' or s[0] == '\'' and s[-1] == '\'':
            tokens.append("'<STRING>'")
        # elif s.isdigit():
        #     tokens.append("'<NUM>'")
        else:
            tokens.append(s)
    return tokens


def lower_case(lang='python'):
    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/py'
    # data_types = ['train', 'valid', 'test']
    data_types = ['train', 'dev', 'test']
    for data_type in data_types:
        code_filename = os.path.join(dir_path, data_type, 'code.txt')
        nl_filename = os.path.join(dir_path, data_type, 'nl.txt')
        new_code = os.path.join(dir_path, data_type, 'new_code.txt')
        new_nl = os.path.join(dir_path, data_type, 'new_nl.txt')
        with open(code_filename,encoding="utf-8") as code_f, \
                open(nl_filename,encoding="utf-8") as nl_f, \
                    open(new_code,'w') as newC, \
                        open(new_nl, 'w') as newN:
            codes = code_f.readlines()
            nls = nl_f.readlines()
            codes = [eval(item).strip() for item in tqdm(codes)]
            codes = [clean_code(item) for item in tqdm(codes)]

            if lang == 'java':
                codes = [tokenize_source_code(item) for item in tqdm(codes)]
                codes = [lower_case_str_arr(item) for item in tqdm(codes)]
                codes = [' '.join(code_tokens_replace_str_num(item)) for item in tqdm(codes)]
            # if lang == 'python':
            #     codes = [python_tokenize(item) for item in tqdm(codes)]
            nls = [split_by_whitespace(item) for item in nls]
            nls = [' '.join(lower_case_str_arr(item)) for item in nls]
            if lang == 'java':
                for code in codes:
                    newC.write(code+'\n')
            if lang == 'python':
                json.dump(codes, newC)
            for nl in nls:
                newN.write(nl)

def extract_dataflow(code, parser, lang, withdfg):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        ast_dict = {}
        index_to_code = {}
        ast_dict[0] = root_node.type
        ast = []
        start_node = []
        type_node = set()
        tree_to_token_index(root_node, ast_dict, index_to_code, ast, 0, start_node, type_node)

        # identifier_dict = {}
        # changeidentifier(root_node, ast_dict, index_to_code, ast, 0, start_node, type_node, identifier_dict)

        reverse_ast = [edge.split()[1] + ' ' + edge.split()[0] for edge in ast]
        ast += reverse_ast
        ast = [edge + ' 1' for edge in ast]
        if withdfg:
            try:
                DFG, _ = parser[1](root_node, index_to_code, {})
            except:
                DFG = []
            DFG = sorted(DFG, key=lambda x: x[1])
            num_ast_edges = len(ast)
            num_dfg_edges = 0
            for d in DFG:
                num_dfg_edges += len(d[-1])
            if num_dfg_edges != 0:
                weight = num_ast_edges // num_dfg_edges
            for d in DFG:
                for start_point in d[-1]:
                    ast.append(str(start_point) + ' ' + str(d[1]) + ' ' + str(weight))
    except:
        ast = []
        ast_dict = {}
        start_node = []
        type_node = set()
    return ast, ast_dict, start_node, type_node

def plot_embeddings(embeddings):
    key_list = []
    emb_list = []
    for key, values in embeddings.items():
        key_list.append(key)
        emb_list.append(values)
    emb_list = np.array(emb_list)

    tsne = TSNE(n_components=2, init='pca', random_state=501)
    node_pos = tsne.fit_transform(emb_list)
    x_min, x_max = np.min(node_pos, 0), np.max(node_pos, 0)
    node_pos = (node_pos - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))

    for i in range(node_pos.shape[0]):
        plt.text(node_pos[i, 0], node_pos[i, 1], key_list[i], color='r')
    plt.show()

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang], AST]
    parsers[lang] = parser

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 path_ids,
                 target_ids,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.path_ids = path_ids
        self.target_ids = target_ids

def sentence2id(tokenizer, max_source_length=256, max_path_len=40, lang='java', withdfg=True):
    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/' + lang
    # dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/CodeXGLUE/Code-Text/dataset/' + lang
    with open(os.path.join(dir_path, 'special_tokens.jsonl')) as f:
        special_nodes = json.load(f)
    tokenizer.add_special_tokens(special_nodes)
    if 'pythonJava' in dir_path:
        data_types = ['train', 'dev', 'test']
    else:
        data_types = ['train', 'valid', 'test']
    for data_type in data_types:
        if withdfg:
            filename = os.path.join(dir_path, data_type + '_paths_wostop.jsonl')
        else:
            filename = os.path.join(dir_path, data_type + '_paths_wodfg_wostop.jsonl')

        # filename = os.path.join(dir_path, data_type + '_paths_random.jsonl')

        with open(filename) as f:
            all_sentences = json.load(f)
        for idx, sentences in enumerate(tqdm(all_sentences)):
            sentences = sentences[:max_source_length]
            for index, sentence in enumerate(sentences):
                tokens = tokenizer.tokenize(sentence)[:max_path_len]
                sentences[index] = tokenizer.convert_tokens_to_ids(tokens)
                if len(tokens) < max_path_len:
                    sentences[index] += (max_path_len-len(tokens)) * [tokenizer.pad_token_id]
            if len(sentences) < max_source_length:
                padding_length = max_source_length - len(sentences)
                sentences += [[tokenizer.sep_token_id] + (max_path_len-1) * [tokenizer.pad_token_id]]
                sentences += (padding_length-1) * [[tokenizer.pad_token_id] + [tokenizer.sep_token_id] + (max_path_len-2) * [tokenizer.pad_token_id]]
            all_sentences[idx] = sentences
        all_sentences = np.array(all_sentences, dtype=np.uint16)
        print(all_sentences.shape)
        if withdfg:
            write_file = os.path.join(dir_path, data_type + '_paths_wostop')
        else:
            write_file = os.path.join(dir_path, data_type + '_paths_wodfg_wostop')

        # write_file = os.path.join(dir_path, data_type + '_paths_random')

        np.save(write_file, all_sentences)

def del_():
    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/java'
    data_types = ['train', 'dev', 'test']
    for data_type in data_types:
        code_filename = os.path.join(dir_path, data_type, 'code.txt')
        nl_file = os.path.join(dir_path, data_type, 'nl.txt')
        write_file = os.path.join(dir_path, data_type+'.jsonl')
        all_rec = []
        with open(code_filename, encoding="utf-8") as cf, open(nl_file, encoding="utf-8") as nf, open(write_file, 'w',encoding="utf-8") as wf:
            code_lines = cf.readlines()
            nl_lines = nf.readlines()
            for code_line, nl_line in zip(code_lines, nl_lines):
                code_line = code_line.strip()
                nl_line = nl_line.strip()
                all_rec.append({'code':code_line, 'summary':nl_line})
            json.dump(all_rec, wf)
def my_plot_bar():
    labels = ['BLEU-J', 'ROUGE-J', 'METEOR-J', 'BLEU-P', 'ROUGE-P', 'METEOR-P']
    num_list = [40.52, 49.46, 34.49, 24.83, 35.04, 16.53]
    num_list1 = [39.13, 48.20, 32.17, 23.94, 34.47, 15.04]
    x = np.arange(len(num_list))
    total_width, n = 0.8, 2
    width = total_width / n
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, num_list1, width, label='Base model', fc='moccasin')
    ax.bar(x + width / 2, num_list, width, label='Our model', fc='orange')
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(ymin=12, ymax=50)
    plt.title('Performance Comparison in Extreme Scenarios')
    ax.legend()
    # plt.show()
    plt.savefig('/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/changevar.svg')
def my_plot_plot():
    y = np.load(
        '/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/y.npy')
    y[4] = 0.44
    _0 = np.load(
        '/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/result0.npy')
    _1 = np.load(
        '/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/result1.npy')
    _2 = np.load(
        '/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/result2.npy')
    _3 = np.load(
        '/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/result3.npy')
    _4 = np.load(
        '/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/result4.npy')
    y_new = np.array([_0, _1, _2, _3, _4])
    x = np.arange(5)
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, y, 'or-.', label='Base model')
    plt.plot(x, y_new, 'ob-.', label='Our model')
    plt.title('Average Cosine Similarity between Randomly Sampled Words')
    plt.xlabel('Layer index')
    plt.legend()
    plt.savefig(
        '/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/cos_sim.svg')
def cal_cossim():
    source_ids = np.load('/data/yzzhao/pythonCode/Code_Intelligence/CodeBERT-master/UniXcoder/downstream-tasks/code-summarization/saved_models/stop_4_Bina_contrast_avgEmb_trans_copy/sample_ids.npy')
    source_ids = torch.Tensor(source_ids).long().cuda()
    source_mask = source_ids.eq(1)
    source_embedding = self.embedding(source_ids)
    sample_input0_emb = source_embedding.detach().cpu()
    cos_sim = np.zeros((5000,5000))
    for i in range(5000):
        i_row = sample_input0_emb[i].repeat(5000, 1)
        i_row_sim = F.cosine_similarity(i_row, sample_input0_emb)
        cos_sim[i] = i_row_sim
    result0 = np.triu(cos_sim, 0).sum() / 12497500

def count_meaningless_token(lang):
    from nltk.corpus import stopwords
    import keyword
    nl_stop_words = set(stopwords.words('english'))
    print(len(nl_stop_words))
    # stop_words.update(keyword.kwlist)
    # print(len(stop_words))
    # stop_words.update(set(['(', ')', '+', '-', '*', '/', '.', '{', '}', '*', '~', '!', '@', '#', '$', '%', '^', '&', '*', '[', ']', '|', '\\', ';', ':', '"', '\'', '?', '/', '<', '>', ',', '_', '`']))
    # stop_words.update(set(['(', ')', '.', '{', '}', '#', '$', '[', ']', '\\', ';', ':', '"', '\'', '?', ',', '_', '`', '\t', '\n']))
    code_stop_word = set(['(', ')', '.', '{', '}', '#', '$', '[', ']', '\\', ';', ':', '"', '\'', '?', ',', '_', '`', '\t', '\n'])
    print(len(code_stop_word))
    # print(stop_words)
    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/CodeXGLUE/Code-Text/dataset/' + lang
    # dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/' + lang
    if 'pythonJava' in dir_path:
        data_types = ['train', 'dev', 'test']
    else:
        data_types = ['train', 'valid', 'test']
    parser = parsers[lang]
    token_nodes = 0
    token_nodes_meanless = 0
    nl_nodes = 0
    nl_nodes_meanless = 0
    for data_type in data_types:
        if 'pythonJava' in dir_path:
            filename = os.path.join(dir_path, data_type, 'code.txt')
            nl_filename = os.path.join(dir_path, data_type, 'nl.txt')
        else:
            filename = os.path.join(dir_path, data_type + '.jsonl')
        with open(filename, encoding="utf-8") as f:
            codes = f.readlines()
            for idx, line in enumerate(tqdm(codes)):
                if 'pythonJava' in dir_path:
                    if lang == 'python':
                        line = eval(line)
                    code = clean_code(line).strip()
                    code = code.replace('\t', '    ')
                else:
                    js = json.loads(line)
                    if 'idx' not in js:
                        js['idx'] = idx
                    code = js['code']
                    nl = js['docstring']
                ast, ast_dict, start_node, type_node = extract_dataflow(code, parser, lang, False)
                start_node = [ast_dict[node] for node in start_node]
                CamelSplitted = []
                for name in start_node:
                    CamelSplitted += re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
                snake_splitted = []
                for name in CamelSplitted:
                    snake_splitted += [word for word in name.split('_')]
                code_len = len(snake_splitted)
                code_meanless_len = len([word for word in snake_splitted if word in code_stop_word])
                nl_token = nl.split()
                nl_len = len(nl_token)
                nl_meanless_len = len([word for word in nl_token if word in nl_stop_words])
                token_nodes += code_len
                token_nodes_meanless += code_meanless_len
                nl_nodes += nl_len
                nl_nodes_meanless += nl_meanless_len
    print(token_nodes)
    print(token_nodes_meanless)
    print(nl_nodes)
    print(nl_nodes_meanless)

def count_unique_tokens(lang='python'):
    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/' + lang
    if 'pythonJava' in dir_path:
        data_types = ['train', 'dev', 'test']
    else:
        data_types = ['train', 'valid', 'test']
    parser = parsers[lang]
    token_nodes = set()
    nl_nodes = set()
    for data_type in data_types:
        datatype_node = set()
        datatype_nl_node = set()
        if 'pythonJava' in dir_path:
            filename = os.path.join(dir_path, data_type, 'code.txt')
            nl_filename = os.path.join(dir_path, data_type, 'nl.txt')
        else:
            filename = os.path.join(dir_path, data_type + '.jsonl')
        with open(filename, encoding="utf-8") as f, open(nl_filename, encoding="utf-8") as nf:
            codes = f.readlines()
            nls = nf.readlines()
            for idx, line in enumerate(tqdm(codes)):
                if 'pythonJava' in dir_path:
                    if lang == 'python':
                        line = eval(line)
                    code = clean_code(line).strip()
                    code = code.replace('\t', '    ')
                else:
                    js = json.loads(code)
                    if 'idx' not in js:
                        js['idx'] = idx
                    code = js['code']
                nl = nls[idx].strip()
                ast, ast_dict, start_node, type_node = extract_dataflow(code, parser, lang, False)
                start_node = [ast_dict[node] for node in start_node]
                CamelSplitted = []
                for name in start_node:
                    CamelSplitted += re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
                snake_splitted = []
                for name in CamelSplitted:
                    snake_splitted += [word for word in name.split('_')]
                datatype_node.update(set(snake_splitted))
                datatype_nl_node.update(set(nl.split()))
            token_nodes.update(datatype_node)
            nl_nodes.update(datatype_nl_node)
        print(data_type + ':' + str(len(datatype_node)) + ' ' + str(len(datatype_nl_node)) + ' ' + str(
            len(datatype_node.intersection(datatype_nl_node))))
    print('total:' + str(len(token_nodes)) + ' ' + str(len(nl_nodes)) + ' ' + str(
        len(token_nodes.intersection(nl_nodes))))

def change_var(tokenizer, max_source_length=256, max_path_len=40, lang='java', withdfg=True, walk_length=20, num_walks=1, p=2, q=0.5):
    example_code = "def sina_xml_to_url_list(xml_data):\n    \"\"\"str->list\n    Convert XML to URL List.\n    From Biligrab.\n    \"\"\"\n    rawurl = []\n    dom = parseString(xml_data)\n    for node in dom.getElementsByTagName('durl'):\n        url = node.getElementsByTagName('url')[0]\n        rawurl.append(url.childNodes[0].data)\n    return rawurl"
    example_txt_path = './example.txt'
    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/CodeXGLUE/Code-Text/dataset/' + lang

    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/' + lang
    if 'pythonJava' in dir_path:
        data_types = ['train', 'dev', 'test']
    else:
        data_types = ['train', 'valid', 'test']
    parser = parsers[lang]
    type_nodes = set()
    code_stop_word = set(
        ['(', ')', '.', '{', '}', '#', '$', '[', ']', '\\', ';', ':', '"', '\'', '?', ',', '_', '`', '\t', '\n'])
    for data_type in data_types:
        if 'pythonJava' in dir_path:
            filename = os.path.join(dir_path, data_type, 'code.txt')
        else:
            filename = os.path.join(dir_path, data_type + '.jsonl')
        all_sentences = []
        with open(filename, encoding="utf-8") as f:
            codes = f.readlines()
            for idx, line in enumerate(tqdm(codes)):
                if 'pythonJava' in dir_path:
                    if lang == 'python':
                        line = eval(line)
                    code = clean_code(line).strip()
                    code = code.replace('\t', '    ')
                else:
                    js = json.loads(code)
                    if 'idx' not in js:
                        js['idx'] = idx
                    code = js['code']
                ast, ast_dict, start_node, type_node = extract_dataflow(code, parser, lang, withdfg)
                type_nodes.update(type_node)

                start_node_string = [ast_dict[node] for node in start_node]
                start = []
                for (id, string) in zip(start_node, start_node_string):
                    if string not in code_stop_word:
                        start.append(id)
                # sentences = []
                # for idx in range(len(start_node)):
                #     sentences.append(' '.join([ast_dict[start_node[idx]]]+random.choices(list(ast_dict.values()), k=walk_length-1)))

                graph_ast = nx.parse_edgelist(ast, create_using=nx.DiGraph(), nodetype=int,
                                              data=[('weight', int)])
                node2vec = Node2Vec(graph_ast, start_node=start, walk_length=walk_length,
                                    num_walks=num_walks, p=p, q=q, workers=1, use_rejection_sampling=0)
                sentences = node2vec.sentences
                from operator import itemgetter
                for idx_i, sentence in enumerate(sentences):
                    sentences[idx_i] = ' '.join(
                        [word for word in itemgetter(*sentence)(ast_dict) if word not in code_stop_word])
                all_sentences.append(sentences)
        if withdfg:
            write_file = os.path.join(dir_path, data_type + '_paths_wostop_changevar.jsonl')
        else:
            write_file = os.path.join(dir_path, data_type + '_paths_wodfg_wostop.jsonl')

        # write_file = os.path.join(dir_path, data_type + '_paths_random.jsonl')

        with open(write_file, 'w') as file_obj:
            json.dump(all_sentences, file_obj)
    type_nodes_file = os.path.join(dir_path, 'special_tokens.jsonl')
    if not os.path.exists(type_nodes_file):
        type_nodes = {'additional_special_tokens': list(type_nodes)}
        with open(type_nodes_file, 'w') as sp_file:
            json.dump(type_nodes, sp_file)
    sentence2id(tokenizer, max_source_length, max_path_len, lang, withdfg)



def generate_ast_with_dfg_and_save_file(tokenizer, max_source_length=256, max_path_len=40, lang='java', withdfg=True, walk_length=20, num_walks=1, p=2, q=0.5):
    example_code = "def sina_xml_to_url_list(xml_data):\n    \"\"\"str->list\n    Convert XML to URL List.\n    From Biligrab.\n    \"\"\"\n    rawurl = []\n    dom = parseString(xml_data)\n    for node in dom.getElementsByTagName('durl'):\n        url = node.getElementsByTagName('url')[0]\n        rawurl.append(url.childNodes[0].data)\n    return rawurl"
    example_txt_path = './example.txt'
    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/CodeXGLUE/Code-Text/dataset/' + lang

    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/' + lang
    dir_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/' + lang
    if 'pythonJava' in dir_path:
        data_types = ['train', 'dev', 'test']
    else:
        data_types = ['train', 'valid', 'test']
    parser = parsers[lang]
    type_nodes = set()
    code_stop_word = set(['(', ')', '.', '{', '}', '#', '$', '[', ']', '\\', ';', ':', '"', '\'', '?', ',', '_', '`', '\t', '\n'])
    for data_type in data_types:
        if 'pythonJava' in dir_path:
            filename = os.path.join(dir_path, data_type, 'code.txt')
        else:
            filename = os.path.join(dir_path, data_type + '.jsonl')
        all_sentences = []
        with open(filename, encoding="utf-8") as f:
            codes = f.readlines()
            for idx, line in enumerate(tqdm(codes)):
                if 'pythonJava' in dir_path:
                    if lang == 'python':
                        line = eval(line)
                    code = clean_code(line).strip()
                    code = code.replace('\t', '    ')
                else:
                    js = json.loads(code)
                    if 'idx' not in js:
                        js['idx']=idx
                    code = js['code']
                ast, ast_dict, start_node, type_node = extract_dataflow(code, parser, lang, withdfg)
                type_nodes.update(type_node)

                start_node_string = [ast_dict[node] for node in start_node]
                start = []
                for (id, string) in zip(start_node, start_node_string):
                    if string not in code_stop_word:
                        start.append(id)
                # sentences = []
                # for idx in range(len(start_node)):
                #     sentences.append(' '.join([ast_dict[start_node[idx]]]+random.choices(list(ast_dict.values()), k=walk_length-1)))

                graph_ast = nx.parse_edgelist(ast, create_using=nx.DiGraph(), nodetype=int,
                                              data=[('weight', int)])
                node2vec = Node2Vec(graph_ast, start_node=start, walk_length=walk_length,
                                    num_walks=num_walks, p=p, q=q, workers=1, use_rejection_sampling=0)
                sentences = node2vec.sentences
                from operator import itemgetter
                for idx_i, sentence in enumerate(sentences):
                    sentences[idx_i] = ' '.join([word for word in itemgetter(*sentence)(ast_dict) if word not in code_stop_word])
                all_sentences.append(sentences)
        if withdfg:
            write_file = os.path.join(dir_path, data_type + '_paths_wostop.jsonl')
        else:
            write_file = os.path.join(dir_path, data_type + '_paths_wodfg_wostop.jsonl')

        # write_file = os.path.join(dir_path, data_type + '_paths_random.jsonl')

        with open(write_file, 'w') as file_obj:
            json.dump(all_sentences, file_obj)
    type_nodes_file = os.path.join(dir_path, 'special_tokens.jsonl')
    if not os.path.exists(type_nodes_file):
        type_nodes = {'additional_special_tokens': list(type_nodes)}
        with open(type_nodes_file, 'w') as sp_file:
            json.dump(type_nodes, sp_file)
    sentence2id(tokenizer, max_source_length, max_path_len, lang, withdfg)


        # with open(os.path.join(dir_path, 'special_tokens.jsonl'), 'w') as sp:
        #     json.dump(type_nodes, sp)
        # with open(write_file, 'w') as file_obj:
        #     json.dump(all_sentences, file_obj)
        # G = nx.read_edgelist(example_txt_path,
        #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
        # model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
        # model = Node2Vec(G, walk_length=10, num_walks=80,
        #                  p=0.25, q=4, workers=1, use_rejection_sampling=0)
        # model.train(window_size=5, iter=3)
        # embeddings = model.get_embeddings()
        # plot_embeddings(embeddings)
        # options = {
        #     'node_color': 'red',
        #     'node_size': 70,
        #     'linewidths': 0,
        #     'width': 0.1,
        #     'with_labels' : True,
        #     'font_size' : 5
        # }
        # nx.draw(G, **options)
        # plt.show()
        # print(example_code)
        # print(code_tokens)
        # print(dfg)

def test():
    from my_model import my_model
    from transformers import get_linear_schedule_with_warmup, AdamW
    model = my_model(50000, d_model=768, hidden_size=768, nhead=8, beam_size=10,
                     max_length=80, sos_id=19, max_source_len = 256, num_layers=4,
                     eos_id=1)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.96},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8, correct_bias=True)
    num_train_epochs, train_dataloader=200, 50000
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_dataloader * num_train_epochs * 0.1),
                                                num_training_steps=train_dataloader * num_train_epochs)
    for epoch in range(num_train_epochs):
        print('step ' + str(epoch) + ', lr: ' + str(scheduler.get_last_lr()))
        for idx in range(train_dataloader):
            scheduler.step()

def get_mean_and_variance(transition_matrix):
    import datetime, time
    start = time.time()
    row, col = transition_matrix.shape
    assert(row == col)
    dimension = row
    ones = np.ones(shape = (dimension, 1))
    eps = 1.0e-16
    expectation = ones
    variance = np.zeros(shape = (dimension, 1))
    # variance = csr_matrix(([0] * dimension, (row_indices, col_indices)), shape = (dimension, 1))
    power = ones
    expansion_order = 10000
    counter = 0
    while counter < expansion_order:
        counter += 1
        power = np.matmul(transition_matrix, power)
        expectation += power
        variance += counter * power
        error = np.linalg.norm(counter * power)
        if True and counter % 500 == 0:
            print("Counter = " + str(counter) + ", error = " + str(error) + ", time = " + str(datetime.datetime.now()))
        if error < eps:
            break
    print("Counter = " + str(counter) + ", error = " + str(error))
    variance = 2.0 * variance
    variance = variance + expectation - np.multiply(expectation, expectation)
    end = time.time()
    print("Total time used in get_mean_and_variance = " + str(end - start) + " seconds. ")
    # print(expectation)
    # return np.matmul(np.linalg.inv(np.eye(dimension) - transition_matrix), np.ones(shape = (dimension, 1))), variance
    return expectation, variance


def compute_hitting_time(code, target, withdfg=True):
    lang = 'python'
    parser = parsers[lang]
    ast, ast_dict, start_node, type_node = extract_dataflow(code, parser, lang, withdfg)
    graph_ast = nx.parse_edgelist(ast, create_using=nx.DiGraph(), nodetype=int,
                                  data=[('weight', int)])
    node2vec = Node2Vec(graph_ast, start_node=start_node, walk_length=20,
                        num_walks=1, p=2, q=0.5, workers=1, use_rejection_sampling=0)
    sentences = node2vec.sentences
    # sentences = sorted(sentences, key=lambda x: x[0])
    for idx_i, sentence in enumerate(sentences):
        for idx_j, word in enumerate(sentence):
            sentences[idx_i][idx_j] = ast_dict[word]
    unnormal_trans_matrix = nx.to_numpy_array(graph_ast)
    trans_matrix = unnormal_trans_matrix / unnormal_trans_matrix.sum(1).reshape((-1,1))
    init_s = np.eye(trans_matrix.shape[0])[19]
    epsilon = 1
    while epsilon > 10e-9:
        next_s = np.dot(init_s, trans_matrix)
        epsilon = np.sqrt(np.sum(np.square(next_s - init_s)))
        init_s = next_s
    print(init_s)
    trans_matrix = np.delete(trans_matrix, target, axis=0)
    trans_matrix = np.delete(trans_matrix, target, axis=1)
    # trans_matrix = np.array([[0,1/2,1/2,0],[1/2,0,1/2,0],[1/3,1/3,0,0],[0,0,0,0]])
    expectation, variance = get_mean_and_variance(trans_matrix)
    print(expectation)
    print(variance)
    return expectation, variance

def proportion():
    import json
    import pandas as pd
    from pathlib import Path
    pd.set_option('max_colwidth', 300)
    from pprint import pprint
    columns_long_list = ['repo', 'path', 'url', 'code',
                         'code_tokens', 'docstring', 'docstring_tokens',
                         'language', 'partition']

    columns_short_list = ['code_tokens', 'docstring_tokens',
                          'language', 'partition']

    def jsonl_list_to_dataframe(file_list, columns=columns_long_list):
        """Load a list of jsonl.gz files into a pandas DataFrame."""
        return pd.concat([pd.read_json(f,
                                       orient='records',
                                       compression='gzip',
                                       lines=True)[columns]
                          for f in file_list], sort=False)
    python_files = sorted(Path('../resources/data/python/').glob('**/*.gz'))
    pydf = jsonl_list_to_dataframe(python_files)



if __name__ == '__main__':
    # main()
    # get_vocab('/data/yzzhao/pythonCode/Code_Intelligence/data/pythonJava/java')
    # lower_case('python')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='microsoft/unixcoder-base-nine', type=str,
                        help="Path to pre-trained tokenizer: e.g. roberta-base")
    parser.add_argument("--max_path_length", default=40, type=int,
                        help="The maximum total path sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--walk_length", default=40, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_walks", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--p", default=2.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--q", default=0.5, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--lang", default='python', type=str,
                        help="Max gradient norm.")
    parser.add_argument("--withdfg", action='store_true',
                        help="Max gradient norm.")

    # args = parser.parse_args()
    # print(args)
    # tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    # count_unique_tokens(lang='python')
    # count_meaningless_token(lang='python')
    # change_var(tokenizer=tokenizer, max_source_length=args.max_source_length,
    #                                     max_path_len=args.max_path_length, lang=args.lang, withdfg=args.withdfg,
    #                                     walk_length=args.walk_length, num_walks=args.num_walks,
    #                                     p=args.p, q=args.q)

    # generate_ast_with_dfg_and_save_file(tokenizer=tokenizer, max_source_length=args.max_source_length,
    #                                     max_path_len=args.max_path_length, lang=args.lang, withdfg=args.withdfg,
    #                                     walk_length=args.walk_length, num_walks=args.num_walks,
    #                                     p=args.p, q=args.q)

    # del_()
    # generate_path(lang='java')
    # test()
    # code1 = 'int countS(String target, ArrayList<String> array){\n    int count = 0;\n    for (String str : array) {\n        if (target.equals (str)) {\n            count++;\n        }\n    }\n    return count;\n}'
    # parser = parsers['java']
    # ast, ast_dict, start_node, type_node = extract_dataflow(code1, parser, lang, True)
    # compute_hitting_time(code1, target=53)
    code2 = 'def factorial (num):\n    result = 1\n    for i in range(1, num + 1):\n        result = result * i\n    return result'
    parser = parsers['python']
    ast, ast_dict, start_node, type_node = extract_dataflow(code2, parser, 'python', True)
    compute_hitting_time(code2, target=53)
    # compute_hitting_time(code='def sum_f(array):\n    sum=0\n    for value in array:\n        sum+=value\n    return sum', target=29, withdfg=True)
    # compute_hitting_time(code = 'def add(a):\n    res=a+1\n    return res\n', target=19)