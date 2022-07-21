"""
The perplexity module contains functions in order to calculate pseudo-perplexity.

Perplexity is a common intrinsic metric for evaluating language models,
it is defined as the exponential average negative log-likelihood of a sequence.
However, for masked language models like BERT, there exists an adaptation called pseudo-perplexity,
which is lower when a language model can 'predict' a given text better.
Since we use these kind of language models in some of our measures, this can be used to help monitor them.
Please refer to this paper for more details: https://doi.org/10.48550/arXiv.1910.14659
"""
import torch
import numpy as np
import pandas as pd

from ..utils import utils

def PPPL_score(GPT2_LM,content):
    content = utils.convert_text_to_string(content)
    tokenize_input = GPT2_LM["tokenizer"].encode(content)
    tensor_input = torch.tensor([tokenize_input[:GPT2_LM["max_length"]]])
    loss = GPT2_LM["model"](tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())
