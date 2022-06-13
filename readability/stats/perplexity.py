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

    def remove_outliers(self,corpus,perplex,stddevratio = 1):
        """
        Outputs a corpus, after removing texts which are considered to be "outliers",
        A text is an outlier if its pseudo-perplexity value is lower or higher than this : mean +- standard_deviation * ratio
        In order to exploit this new corpus, you'll need to make a new Readability instance.
        For instance : new_r = Readability(r.remove_outliers(r.perplexity(),1))

        :return: a corpus, in a specific format where texts are represented as lists of sentences, which are lists of words.
        :rtype: dict[class][text][sentence][token]
        """
        levels = list(perplex.keys())
        moy_ppl= list()
        for level in levels:
            moy=0
            for score in perplex[level]:
                moy+= score/len(perplex[level])
            moy_ppl.append(moy)

        stddev_ppl = list()
        for index, level in enumerate(levels):
            stddev=0
            for score in perplex[level]:
                stddev += ((score-moy_ppl[index])**2)/len(perplex[level])
            stddev = math.sqrt(stddev)
            stddev_ppl.append(stddev)

        outliers_indices = perplex.copy()
        for index, level in enumerate(levels):
            outliers_indices[level] = [idx for idx in range(len(perplex[level])) if perplex[level][idx] > moy_ppl[index] + (stddevratio * stddev_ppl[index]) or perplex[level][idx] < moy_ppl[index] - (stddevratio * stddev_ppl[index])]
            print("nb textes enleves(",level,") :", len(outliers_indices[level]),sep="")
            print(outliers_indices[level])

        corpus_no_outliers = copy.deepcopy(corpus)
        for level in levels:
            offset = 0
            for index in outliers_indices[level][:]:
                corpus_no_outliers[level].pop(index - offset)
                offset += 1
            print("New number of texts for class", level, ":", len(corpus_no_outliers[level]))
        print("You have to make a new Readability instance to use this new corpus.")
        return corpus_no_outliers

pppl_calculator = PPPL_calculator()
