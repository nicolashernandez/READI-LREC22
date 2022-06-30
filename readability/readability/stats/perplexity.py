"""The perplexity module contains functions in order to calculate pseudo-perplexity"""
import torch
import numpy as np
import math
import copy
from ..utils import utils

def PPPL_score(GPT2_LM,content):
    content = utils.convert_text_to_string(content)
    tokenize_input = GPT2_LM["tokenizer"].encode(content)
    tensor_input = torch.tensor([tokenize_input[:GPT2_LM["max_length"]]])
    loss = GPT2_LM["model"](tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())
