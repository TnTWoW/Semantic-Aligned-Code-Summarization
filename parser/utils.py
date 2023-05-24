import re
from io import StringIO
import  tokenize
def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def changeidentifier(root_node, ast_dict, index_to_code, ast, parent_index, start_node, type_node, identifier_dict):
    for idx, child in enumerate(root_node.children):
        index = len(ast_dict)
        if (len(child.children) == 0 or child.type == 'string') and child.type != 'comment':
            code = child.text.decode()
            if child.type == 'identifier':
                if code in identifier_dict.keys():
                    code = identifier_dict[code]
                else:
                    new_var = chr(ord('a') + len(identifier_dict))
                    identifier_dict[code] = new_var
                    code = new_var
            ast_dict[index] = code
            index_to_code[(child.start_point, child.end_point)] = (index, code)
            if code != '':
                start_node.append(index)
        else:
            ast_dict[index] = child.type
            type_node.add(child.type)
        ast.append(str(parent_index) + ' ' + str(index))
        changeidentifier(child, ast_dict, index_to_code, ast, index, start_node, type_node, identifier_dict)

def tree_to_token_index(root_node, ast_dict, index_to_code, ast, parent_index, start_node, type_node):
    for idx, child in enumerate(root_node.children):
        index = len(ast_dict)
        if (len(child.children) == 0 or child.type == 'string') and child.type != 'comment':
            code = child.text.decode()
            ast_dict[index] = code
            index_to_code[(child.start_point, child.end_point)] = (index, code)
            if code != '':
                start_node.append(index)
        else:
            ast_dict[index] = child.type
            type_node.add(child.type)
        ast.append(str(parent_index) + ' ' + str(index))
    # for idx, child in enumerate(root_node.children):
        tree_to_token_index(child, ast_dict, index_to_code, ast, index, start_node, type_node)
    # index = len(ast_dict)
    # if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
    #     ast_dict[index] = root_node.text.decode()
    #     index_to_code[(root_node.start_point, root_node.end_point)] = (index, root_node.text.decode())
    # else:
    #     ast_dict[index] = root_node.type
    # for child in root_node.children:
    #     tree_to_token_index(child, ast_dict, index_to_code)

def tree_to_token_index_var(root_node):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [(root_node.start_point,root_node.end_point)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_index(child)
        return code_tokens
    
def tree_to_variable_index(root_node,index_to_code):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        index=(root_node.start_point,root_node.end_point)

        _,code=index_to_code[index]
        if root_node.type!=code:
            return [(root_node.start_point,root_node.end_point)]
        else:
            return []
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_variable_index(child,index_to_code)
        return code_tokens    

def index_to_code_token(index,code):
    start_point=index[0]
    end_point=index[1]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]   
    return s
   