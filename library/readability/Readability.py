"""
The Readability module interacts with the library's submodules to provide a bunch of useful functions / reproduce the READI paper's contents

It provides the following services :
At start-up : it "compiles" a text into a structure useful for the other functions, and also calculates relevant statistics (number of words, sentences, syllables, etc..)
Enables a way to access "simple" scores by using these pre-calculated statistics
Perform lazy loading of more complicated things, like calculating perplexity or the use of Machine Learning / Deep Learning models
"""

import pandas as pd
import spacy
import unidecode
from .stats import diversity, perplexity, common_scores
from .methods import *
from .models import *

# Checklist :
#     Remake structure to help differenciate between functions : ~ I think it's okay but I need some feedback
#     Enable a way to "compile" in order to use underlying functions faster : ~ Logic is ready, but the code isn't. 
#     I most likely need to make a Statistics class and give it a bunch of attributes
#     Make sure code works both for texts (strings, or pre-tokenized texts) and corpus structure : ~ It's in progress..
#     Add the methods related to machine learning or deep learning : X
#     Experiment further : X
#     Add other measures that could be useful : X



#TODO : put this elsewhere, it's needed for making text statistics but it's not exactly part of the readability object.
def syllablesplit(input):
    """
    syllablesplit takes as input a word, and estimates the number of syllables in that word
    Updated version of this function can be improved by taking into account the lang attribute from Readability class.
    :param input: Content of a word
    :type input: str

    :return: The estimated number of syllables in a word.
    :rtype: int
    """
    nb_syllabes = 0
    syllables='aeiouy'
    for char in input:
        for syl in syllables:
            if syl == unidecode(char):
                nb_syllabes+=1
                break
    return nb_syllabes


class Readability:
    """
    The Readability class provides a way to access the underlying library modules in order to help estimate the complexity of any given text
    List of methods : __init__, corpus_info, .common_scores, scores
    List of attributes : content, content_type, classes, lang, nlp, perplexity_processor, statistics

    In its current state, scores are only accurate for the French language, but this can change in the future.
    """
    def __init__(self, content, lang = "fr", nlp_name = "spacy_sm", perplexity_processor = "gpt2"):
        """
        Constructor of the Readability class, won't return any value but creates the attributes :
        self.content, self.content_type, self.nlp, self.lang

        :param content: Content of a text, or a corpus
        :type content: str, list(str), list(list(str)), converted into list(list(str)) or dict[class][text][sentence][token]

        :param lang: Language the text was written in, in order to adapt some scores.
        :type lang: str

        :param nlp_name: Type of NLP processor to use, indicated by a "type_subtype" string.
        :type nlp_name: str

        :param perplexity_processor: Type of processor to use for the calculation of pseudo-perplexity
        :type perplexity_processor: str


        """
        #V0 will download everything at once when called.
        #V1 could implement lazy loading for the heavy stuff, like using a transformer model.

        #1) Compile/convert text/corpora into relevant format

        #2 + 3) Prepare statistics into a list or object + Load the "small" or local stuff like spacy (via a compile function)

        #4) Prepare ways to lazy load heavier stuff
        self.lang = lang
        self.perplexity_processor = perplexity_processor


        # Handle the NLP processor (mainly for tokenization in case we're given a text as a string)
        # Note : I tried adding the spacy model as a dependency in setup.cfg:
        # fr_core_news_sm@https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.3.0/fr_core_news_sm-3.3.0.tar.gz#egg=fr_core_news_sm
        # But I can't figure out how to use it, so this is a workaround.
        print("Acquiring Natural Language Processor...")
        if lang == "fr" and nlp_name == "spacy_sm":
            try:
                self.nlp = spacy.load('fr_core_news_sm')
                print("DEBUG: Spacy model location (already installed) : ",self.nlp._path)
            except OSError:
                print('Downloading spacy language model \n'
                            "(Should happen only once)")
                from spacy.cli import download
                download('fr_core_news_sm')
                self.nlp = spacy.load('fr_core_news_sm')
                print("DEBUG: Spacy model location : ",self.nlp._path)
        else:
            self.nlp = None
        

        #Handling text that needs to be converted into lists of tokens
        if isinstance(content, str):
            print("DEBUG: Text recognized as string, converting to list of lists")
            self.content_type = "text"
            self.content = [[token.text for token in sent] for sent in self.nlp(content).sents]
            nb_words = 0
            for sentence in self.content:
                nb_words += len(sentence)
            if nb_words < 101:
                print("WARNING : Text length is less than 100 words, some scores will be inaccurate.")
            print("DEBUG : converted content is :", self.content)

        #Handling text that doesn't need to be converted
        elif any(isinstance(el, list) for el in content):
            print("DEBUG : Text recognized as list of sentences, not converting")
            self.content_type = "text"
            self.content = content
            print("DEBUG: recognized content is :",self.content)

        #Handling text that was only converted into tokens
        elif isinstance(content, list):
            print("DEBUG : Text is already tokenized, converting to a list of sentences")
            self.content_type = "text"
            content = ' '.join(content)
            self.content = [[token.text for token in sent] for sent in self.nlp(content).sents]
            nb_words = 0
            for sentence in self.content:
                for token in sentence:
                    nb_words += len(sentence)
            if nb_words < 101:
                print("WARNING : Text length is less than 100 words, scores will be inaccurate and meaningless.")
            print("DEBUG : converted content is :", self.content)


        #Check if input is a corpus.
        else:
            #Reminder, structure needed is : dict => list of texts => list of sentences => list of words
            #TODO : check this with a bunch of edge cases
            if type(content) == dict:
                if isinstance(content[list(content.keys())[0]], list):
                    if isinstance(content[list(content.keys())[0]][0], list):
                        if isinstance(content[list(content.keys())[0]][0][0], list):
                            print("DEBUG : recognized as corpus")
                            self.content_type = "corpus"
                            self.content = content
                            self.classes = list(content.keys())
        

    #The calculation of common_scores are handled by passing a stats list or object that contains the relevant information.
    #If this informations is unknown, the method will attempt to extract them from the current text.
    #TODO : replace these if elif with try catch
    def gfi(self):
        if self.content_type == "text":
            if hasattr(self, "statistics"):
                print("DEBUG : pre-existing information was found")
                return common_scores.GFI_score(self.content, self.nlp, self.statistics)
            else:
                print("DEBUG : pre-existing information was not found, so GFI SHOULD determine it by itself")
                return common_scores.GFI_score(self.content, self.nlp)
        elif self.content_type == "corpus":
            print("just do a loop and return a list of scores")
        return -1
    def ari(self):
        if self.content_type == "text":
            if hasattr(self, "statistics"):
                print("DEBUG : pre-existing information was found")
                return common_scores.ARI_score(self.content, self.nlp, self.statistics)
            else:
                print("DEBUG : pre-existing information was not found, so ARI SHOULD determine it by itself")
                return common_scores.ARI_score(self.content, self.nlp)
        elif self.content_type == "corpus":
            print("just do a loop and return a list of scores")
        return -1
    def fre(self):
        if self.content_type == "text":
            if hasattr(self, "statistics"):
                print("DEBUG : pre-existing information was found")
                return common_scores.FRE_score(self.content, self.nlp, self.statistics)
            else:
                print("DEBUG : pre-existing information was not found, so FRE SHOULD determine it by itself")
                return common_scores.FRE_score(self.content, self.nlp)
        elif self.content_type == "corpus":
            print("just do a loop and return a list of scores")
        return -1
    def fkgl(self):
        if self.content_type == "text":
            if hasattr(self, "statistics"):
                print("DEBUG : pre-existing information was found")
                return common_scores.FKGL_score(self.content, self.nlp, self.statistics)
            else:
                print("DEBUG : pre-existing information was not found, so FKGL SHOULD determine it by itself")
                return common_scores.FKGL_score(self.content, self.nlp)
        elif self.content_type == "corpus":
            print("just do a loop and return a list of scores")
        return -1
    def smog(self):
        if self.content_type == "text":
            if hasattr(self, "statistics"):
                print("DEBUG : pre-existing information was found")
                return common_scores.SMOG_score(self.content, self.nlp, self.statistics)
            else:
                print("DEBUG : pre-existing information was not found, so SMOG SHOULD determine it by itself")
                return common_scores.SMOG_score(self.content, self.nlp)
        elif self.content_type == "corpus":
            print("just do a loop and return a list of scores")
        return -1
    def rel(self):
        if self.content_type == "text":
            if hasattr(self, "statistics"):
                print("DEBUG : pre-existing information was found")
                return common_scores.REL_score(self.content, self.nlp, self.statistics)
            else:
                print("DEBUG : pre-existing information was not found, so REL SHOULD determine it by itself")
                return common_scores.REL_score(self.content, self.nlp)
        elif self.content_type == "corpus":
            print("just do a loop and return a list of scores")
        return -1
    def scores(self):
        #return an optimized version of the previous scores.
        #Calling each function implies making an if branch for every text encountered. Profiling is needed to see if this has an impact on performance or not
        #TODO : improve further by giving the self.corpus_statistics once we figure out how to make it.
        return common_scores.traditional_scores(self.content)


    def compile(self):
        """
        Calculates a bunch of stuff to make some underlying functions faster.
        """
        if self.content_type == "text":
            totalWords = 0
            nbLongWords = 0
            totalSentences = len(self.content)
            totalCharacters = 0
            totalSyllables = 0
            nbPolysyllables = 0
            for sentence in self.content:
                totalWords += len(sentence)
                nbLongWords += len([token for token in sentence if len(token)>6])
                totalCharacters += sum(len(token) for token in sentence)
                totalSyllables += sum(syllablesplit(word) for word in sentence)
                nbPolysyllables += sum(1 for word in sentence if syllablesplit(word)>=3)
            #TODO : put these in a Statistics class or something, and let that class be used as an attribute for Readability.
            return 0

        elif self.content_type == "corpus":
            return 0
            #Same thing as above, but assign every statistic to a text, somehow.
            #I suppose we could make a self.corpus_statistics attribute with shape : dict[class][statistics]
        return 0

    #Note : Maybe this should go in the stats subfolder to have less bloat.
    def corpus_info(self):
        """
            Output several basic statistics such as number of texts, sentences, or tokens, alongside size of the vocabulary.
                
            :param corpus: Dictionary of lists of sentences (represented as a list of tokens)
            :type corpus: dict[class][text][sentence][token]

            :return: a pandas dataframe 
            :rtype: pandas.core.frame.DataFrame
        """
        #TODO : make that an exception instead?
        if self.content_type != "corpus":
            print("Current input isn't recognized as a corpus. Please provide a dictionnary with the following format : dict[class][text][sentence][token]")
            return 0
        else:
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
