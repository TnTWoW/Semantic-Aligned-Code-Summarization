# Copyright (c) yzzhao.

from tree_sitter import Language, Parser
from .utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
import re

# def AST(root_node, index_to_code):
#     state = []
#     if root_node.type == 'string':
#         return state
#     for child in root_node.children:
#         if (len(child.children) == 0 or child.type == 'string') and child.type != 'comment':
#             idx, code = index_to_code[(child.start_point, child.end_point)]
#             if len(code.split('_')) > 1 or len(re.findall('[a-zA-Z][^A-Z]*', code)) > 1:  # 下划线分割命名 or 驼峰命名
#                 splited_tokens = [x for x in code.split('_') if x != ''] if len(code.split('_')) > 1 else re.findall('[a-zA-Z][^A-Z]*', code)
#                 for token in splited_tokens:
#                     state.append(root_node.type + '<split>' + token)
#                 for i in range(len(splited_tokens) - 1):
#                     state.append(splited_tokens[i] + '<split>' + splited_tokens[i + 1])
#             else:
#                 state.append(root_node.type+ '<split>'+ code)
#         else:
#             state.append(root_node.type + '<split>' + child.type)
#     for child in root_node.children:
#         state += AST(child, index_to_code)
#     return state

# def AST(root_node, ast, start_id):
#     for child in root_node.children:
#         ast.append(str(start_id) + ' '+ )


def AST(root_node, index_to_code):
    state = []
    if root_node.type == 'string':
        return state
    for child in root_node.children:
        if (len(child.children) == 0 or child.type == 'string') and child.type != 'comment':
            idx, code = index_to_code[(child.start_point, child.end_point)]
            state.append(root_node.type + '<split>'+ str(idx))
        else:
            state.append(root_node.type + '<split>'+ child.type)
    for child in root_node.children:
        state += AST(child, index_to_code)
    return state
