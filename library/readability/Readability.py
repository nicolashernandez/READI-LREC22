"""
The Readability module interacts with the library's modules to provide a bunch of useful functions / reproduce the READI paper's contents

It is meant to provide the following :
At start-up : it "compiles" a text into a structure useful for the other functions, and also calculates relevant statistics (number of words, sentences, syllables, etc..)
Enables a way to access "simple" scores by using these pre-calculated statistics
Perform lazy loading of more complicated things, like calculating perplexity or the use of Machine Learning / Deep Learning models
"""


import pandas as pd
from .stats import *
from .methods import *
from .models import *

class Readability:
    """
    The Readability class provides a way to access the underlying library modules in order to help estimate the complexity of any given text
    List of methods : __init__, corpus_info, common_scores
    List of attributes : content, content_type, lang, nlp_processor, perplexity_processor

    In its current state, scores are only accurate for the French language, but this can change in the future.
    
    """
    def __init__(self, content, lang = "fr", nlp_processor = "spacy_sm", perplexity_processor = "gpt2"):
        print("hello world my first parameter is", lang)
        #V0 will download everything at once when called.
        #V1 could implement lazy loading for the heavy stuff, like using a transformer model.
        
        #Handling text that needs to be converted
        if type(content) == str:
            print("amogus is a simple string")
            self.content_type = "text"
            self.content = content ; print(self.content)

        #Handling text that doesn't need to be converted
        elif any(isinstance(el, list) for el in content):
            print("sugoma is a list of lists. probably")
            self.content_type = "text"
            self.content = content ; print(self.content)

        #1) Compile text/corpora into relevant format


        #2) Prepare statistics

        #3) Load the "small" or local stuff like spacy"


        #4) Prepare ways to lazy load heavier stuff

        #Suppose input is a corpus.
        else:
            #Reminder, structured needed is : dict => list of texts => list of sentences => list of words
            if type(content) == dict:
                if isinstance(content[list(content.keys())[0]], list):
                    if isinstance(content[list(content.keys())[0]][0], list):
                        if isinstance(content[list(content.keys())[0]][0][0], list):
                            print(content[list(content.keys())[0]][0][0])
                            self.content_type = "corpus"
                            self.content = content
                            self.classes = list(content.keys())
    def corpus_info(self):
        if self.content_type == "corpus":
            """
            Output several basic statistics such as number of texts, sentences, or tokens, alongside size of the vocabulary.
                
            :param corpus: Dictionary of lists of sentences (represented as a list of tokens)
            :type corpus: dict[class][text][sentence][token]

            :return: a pandas dataframe 
            :rtype: pandas.core.frame.DataFrame
            """

            # Extract the classes from the dictionary's keys.
            corpus = self.content
            levels = self.classes
            #TODO : Need to check that corpus is just a reference to self.content and not a copy, easier for coding, but might not be optimized
            cols = levels + ['total']

            # Build vocabulary
            vocab = dict()
            for level in levels:
                vocab[level] = list()
                for text in corpus[level]:
                    unique = set()
                    for sent in text:
                        #for sent in text['content']:
                        for token in sent:
                            unique.add(token)
                        vocab[level].append(unique)
            
            # Number of texts, sentences, and tokens per level, and on the entire corpus
            nb_ph_moy= list()
            nb_ph = list()
            nb_files = list()
            nb_tokens = list()
            nb_tokens_moy = list()
            len_ph_moy = list()

            for level in levels:
                nb_txt = len(corpus[level])
                nb_files.append(nb_txt)
                nbr_ph=0
                nbr_ph_moy =0
                nbr_tokens =0
                nbr_tokens_moy =0
                len_phr=0
                len_phr_moy=0
                for text in corpus[level]:
                    nbr_ph+=len(text)
                    temp_nbr_ph = len(text)
                    nbr_ph_moy+=len(text)/nb_txt
                    for sent in text:
                        nbr_tokens+=len(sent)
                        nbr_tokens_moy+= len(sent)/nb_txt
                        len_phr+=len(sent)
                    len_phr_moy+=len_phr/temp_nbr_ph
                    len_phr=0
                len_phr_moy = len_phr_moy/nb_txt
                nb_tokens.append(nbr_tokens)
                nb_tokens_moy.append(nbr_tokens_moy)
                nb_ph.append(nbr_ph)
                nb_ph_moy.append(nbr_ph_moy)
                len_ph_moy.append(len_phr_moy)
            nb_files_tot = sum(nb_files)
            nb_ph_tot = sum(nb_ph)
            nb_tokens_tot = sum(nb_tokens)
            nb_tokens_moy_tot = nb_tokens_tot/nb_files_tot
            nb_ph_moy_tot = nb_ph_tot/nb_files_tot
            len_ph_moy_tot = sum(len_ph_moy)/len(levels)
            nb_files.append(nb_files_tot)
            nb_ph.append(nb_ph_tot)
            nb_tokens.append(nb_tokens_tot)
            nb_tokens_moy.append(nb_tokens_moy_tot)
            nb_ph_moy.append(nb_ph_moy_tot)
            len_ph_moy.append(len_ph_moy_tot)

            # Vocabulary size per class
            taille_vocab =list()
            taille_vocab_moy=list()
            all_vocab =set()
            for level in levels:
                vocab_level = set()
                for text in vocab[level]:
                    for token in text:
                        all_vocab.add(token)
                        vocab_level.add(token)
                taille_vocab.append(len(vocab_level))

            taille_vocab.append(len(all_vocab))

            # Mean size of vocabulary
            taille_vocab_moy = list()
            taille_moy_total = 0
            for level in levels:
                moy=0
                for text in vocab[level]:
                    taille_moy_total+= len(text)/nb_files_tot
                    moy+=len(text)/len(vocab[level])
                taille_vocab_moy.append(moy)
            taille_vocab_moy.append(taille_moy_total)  

            # The resulting dataframe can be used for statistical analysis.
            df = pd.DataFrame([nb_files,nb_ph,nb_ph_moy,len_ph_moy,nb_tokens,nb_tokens_moy,taille_vocab,taille_vocab_moy],columns=cols)
            df.index = ["Nombre de fichiers","Nombre de phrases total","Nombre de phrases moyen","Longueur moyenne de phrase","Nombre de tokens", "Nombre de token moyen","Taille du vocabulaire", "Taille moyenne du vocabulaire"]
            return round(df,0)
        else:
            print("current input isn't recognized as a corpus.")
        return 0
