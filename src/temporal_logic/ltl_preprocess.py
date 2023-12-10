import torch
import numpy as np
import re

sltl_tokens = ['and', 'or', 'not', 'next', 'until', 'eventually', 'then', 'True', 'False']

def preprocess(formulas, vocab, device=None):
    # convert formulas (list[str]) into Tensor
    var_indexed_texts = []
    max_text_len = 25

    for formula in formulas:
        formula = str(formula) # transforming the ltl formula into a string
        tokens = re.findall("([a-z]+)", formula.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(formulas), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)

if __name__=="__main__":
    propositions = ['a','b','c','d','n','o']
    vocab = dict()
    n_tokens = 0
    for t in sltl_tokens:
        vocab[t]=n_tokens
        n_tokens+=1
    for t in propositions:
        assert t not in sltl_tokens
        vocab[t]=n_tokens
        n_tokens+=1
    res1 = ('then', ('until', ('not', 'n'), 'c'), ('until', ('not', 'n'), 'o'))
    preprocess([res1], vocab)