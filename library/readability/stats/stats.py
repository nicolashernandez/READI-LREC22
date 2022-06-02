"""
Temporary module for methods/classes that haven't been properly defined yet.
This will probably be divided into several scripts for ease of code/maintaining future versions
Probably something like :
scores
other_scores (text diversity like TTR)
perplexity
"""

import math
import numpy as np
import spacy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

import string
