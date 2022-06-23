"""The perplexity module contains functions in order to calculate pseudo-perplexity"""
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import math
import copy

class PPPL_calculator:
    """
    The PPPL_calculator class provides a way to use a specific language model for calculating pseudo-perplexity
    List of methods : load_model, gpt2_pppl_scores, PPPL_score_text, PPPL_score, remove_outliers
    List of attributes : model, tokenizer, max_length, model_loaded
    """
    def load_model(self, perplexity_processor = None):
        #TODO: change this based on perplexity_processor.
        model_name = "asi/gpt-fr-cased-small"
        # Load pre-trained model (weights)
        with torch.no_grad():
                self.model = GPT2LMHeadModel.from_pretrained(model_name)
                self.model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.max_length = 100
        self.model_loaded = True
        return 0

    def gpt2_pppl_score(self,sentence):
        tokenize_input = self.tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input[:self.max_length]])
        loss=self.model(tensor_input, labels=tensor_input)[0]
        return np.exp(loss.detach().numpy())

    def PPPL_score_text(self,text):
        """Calculates pseudo-perplexity for a text, based on language model used."""
        if isinstance(text, list):
            tex = ''
            for sent in text:
                tex +=' '.join(sent)
                calcul = self.gpt2_pppl_score(tex.strip())
                return calcul
        elif isinstance(text,str):
            calcul = self.gpt2_pppl_score(tex.strip())
            return calcul
        else:
            raise TypeError('Content type is not text, please use part of the corpus to use this function')
    
    def PPPL_score(self,corpus):
        """Calculates pseudo-perplexity for a corpus, based on language model used."""
        levels = list(corpus.keys())
        perplex = dict()
        for level in levels:
            print("Now calculating pseudo-perplexity for class :",level)
            perplex[level] = []
            for text in corpus[level]:
                tex = ''
                for sent in text:
                    tex +=' '.join(sent)
                perplex[level].append(self.gpt2_pppl_score(tex.strip()))
        return perplex

pppl_calculator = PPPL_calculator()
