from utils import *
import os
import json
import pickle


data_path = '/data/yzzhao/pythonCode/Code_Intelligence/data/CodeXGLUE/Code-Text/dataset'

benchmark = ['go', 'java','javascript','php','python','ruby']

types = ['train', 'valid', 'test']


def trans_code_to_graph_txt():
    for language in benchmark:
        for prefix in types:
            count = 0
            filepath = os.path.join(data_path, language, prefix)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filename = filepath + '.jsonl'
            with open(filename, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    js = json.loads(line)
                    if 'idx' not in js:
                        js['idx'] = idx
                    code = js['code']
                    name = os.path.join(filepath, str(count)+'.txt')
                    count += 1
                    parser = parsers[language]
                    code_tokens, dfg = extract_dataflow(code, parser, language)
                    transform(dfg, name)

def read_graph_and_convert_to_vector():
    for language in benchmark:
        for prefix in types:
            filepath = os.path.join(data_path, language, prefix)
            with open(os.path.join(filepath, 'embedding.pkl'), "wb") as tf:
                for f in os.listdir(filepath):
                    filename = os.path.join(filepath, f)
                    G = nx.read_edgelist(filename,
                                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
                    model = DeepWalk(G, walk_length=10, num_walks=10, workers=10)
                    try:
                        model.train(window_size=5, iter=3)
                    except:
                        print(G.nodes())
                        print(G.edges())
                    embeddings = model.get_embeddings()
                    pickle.dump(embeddings, tf)


if __name__ == '__main__':
    # trans_code_to_graph_txt()
    read_graph_and_convert_to_vector()