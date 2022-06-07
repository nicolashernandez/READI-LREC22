"""The perplexity module contains functions in order to calculate pseudo-perplexity"""
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import math

class PPPL_calculator:
    def load_model(self, perplexity_processor):

        #TODO: change this based on perplexity_processor.

        model_name = "asi/gpt-fr-cased-small"
        # Load pre-trained model (weights)
        with torch.no_grad():
                self.model = GPT2LMHeadModel.from_pretrained(model_name)
                self.model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.max_length = 100

        print("Model online, you can now use .PPPL_score()")
        #Apparently I can pass name as a parameter in init
        return 0
    def gpt2_pppl_score(self,sentence):
        tokenize_input = self.tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input[:self.max_length]])
        loss=self.model(tensor_input, labels=tensor_input)[0]
        return np.exp(loss.detach().numpy())
    def PPPL_score_text(self,text):
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
            #return type error
            print("todo: return type error")
            return -1
    #maybe make just one function and change behavior according to type.
    def PPPL_score(self,corpus,save = False):
        levels = list(corpus.keys())
        perplex = dict()
        nb_tot = 0
        for level in levels:
            perplex[level] = []
            ppl = 0
            for text in corpus[level]:
                tex = ''
                for sent in text:
                    tex +=' '.join(sent)
                perplex[level].append(self.gpt2_pppl_score(tex.strip()))
        return perplex

        #with open('perplex_jll.pkl','wb') as file:
        #    pickle.dump(perplex,file)
    def remove_outliers(self,corpus, perplex,stddevratio = 1):
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
            #print(stddev)
            stddev_ppl.append(stddev)

        outliers_indices = perplex.copy()
        for index, level in enumerate(levels):
            outliers_indices[level] = [idx for idx in range(len(perplex[level])) if perplex[level][idx] > moy_ppl[index] + (stddevratio * stddev_ppl[index]) or perplex[level][idx] < moy_ppl[index] - (stddevratio * stddev_ppl[index])]
            print(outliers_indices[level])
            print("nb textes enleves(",level,"):", len(outliers_indices[level]))
        import copy
        corpus_no_outliers = copy.deepcopy(corpus)
        for level in levels:
            offset = 0
            for index in corpus_no_outliers[level][:]:
                corpus_no_outliers[level].pop(index - offset)
                offset += 1
            print("Number of texts for class", level, ":", len(corpus_no_outliers[level]))
        return corpus_no_outliers


pppl_calculator = PPPL_calculator()

# Todo : put a custom error message so that user remembers to do load_model()
# i'd put it in the __init__ but both cases are not optimal
# case one, if creation at import : takes too long for first start, weird behavior, not wanted
# case two, if no creation at import : other functions rely on the pppl, so if they don't have access to it, everything breaks.
# Not too important for now.
