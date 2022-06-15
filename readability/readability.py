"""
The Readability module interacts with the library's submodules to provide a bunch of useful functions / reproduce the READI paper's contents

It provides the following services :
At start-up : it "converts" a text into a structure useful for the other functions, and can also calculate relevant statistics via the .compile function (number of words, sentences, syllables, etc..)
Enables a way to access "simple" scores using these statistics
Perform lazy loading of more complicated things, like calculating perplexity or the use of Machine Learning / Deep Learning models
It can be customized based on which NLP processor to use, or what language to evaluate.
"""
import pandas as pd
import spacy
import utils
from .stats import diversity, perplexity, common_scores
from .methods import methods
from .models import bert, fasttext, models

# Checklist :
#     Remake structure to help differenciate between functions : ~ I think it's okay but I need some feedback
#     Enable a way to "compile" in order to use underlying functions faster : ~ It's done, need to implement tests.
#     Make sure code works both for texts (strings, or pre-tokenized texts) and corpus structure : ~ I think it works now.
#     Add the methods related to machine learning or deep learning : V
#     Add examples to the notebook to show how it can be used : ~ In progress.
#     Add other measures that could be useful (Martinc | Crossley): X
#     Experiment further : X
#     

#10/06/22 checklist :
# Convert demo_ functions into more user-friendly functions, with detailed documentation
# Figure out why some results are slightly off for the ljl corpus compared to what's in the paper
# Start implementing more measures / experiment



# FIXME : several formulas are incorrect, as outlined in the submodule stats/common_scores.
# For now, we kept these as is, in order to keep the paper's experiments reproducible


# NOTE: There probably exists a better way to create an Statistics object as an attribute of Readability.
class Statistics:
    pass

class Readability:
    """
    The Readability class provides a way to access the underlying library modules in order to help estimate the complexity of any given text
    List of methods : __init__, corpus_info, compile, stats, score(score_name), scores, perplexity, remove_outliers, diversity(type,mode)
    List of attributes : content, content_type, lang, nlp, perplexity_processor, classes, statistics, corpus_statistics

    In its current state, scores are meant to be used with the French language, but this can change in the future.
    """
    def __init__(self, content, lang = "fr", nlp = "spacy_sm", perplexity_processor = "gpt2"):
        """
        Constructor of the Readability class, won't return any value but creates the attributes :
        self.content, self.content_type, self.nlp, self.lang, and self.classes only if input is a corpus.

        :param content: Content of a text, or a corpus
        :type content: str, list(str), list(list(str)), converted into list(list(str)) or dict[class][text][sentence][token]

        :param lang: Language the text was written in, in order to adapt some scores.
        :type lang: str

        :param nlp: Type of NLP processor to use, indicated by a "type_subtype" string.
        :type nlp: str

        :param perplexity_processor: Type of processor to use for the calculation of pseudo-perplexity
        :type perplexity_processor: str
        """
        self.lang = lang
        self.perplexity_processor = perplexity_processor
        self.perplexity_calculator = perplexity.pppl_calculator


        # Handle the NLP processor (mainly for tokenization in case we're given a text as a string)
        # Note : I tried adding the spacy model as a dependency in setup.cfg:
        # fr_core_news_sm@https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.3.0/fr_core_news_sm-3.3.0.tar.gz#egg=fr_core_news_sm
        # But I can't figure out how to use it, so this is a workaround.
        print("Acquiring Natural Language Processor...")
        if lang == "fr" and nlp == "spacy_sm":
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
            print("ERROR : Natural Language Processor not found for parameters :  lang=",lang," nlp=",nlp,sep="")
            self.nlp = None
        
        # Handling text that needs to be converted into lists of tokens
        # NOTE: maybe allow removing punctuation by checking token.is_punct as a parameter
        if isinstance(content, str):
            self.content_type = "text"
            self.content = [[token.text for token in sent] for sent in self.nlp(content).sents]
            #self.content = [[token.text for token in sent if not token.is_punct] for sent in self.nlp(content).sents]
            nb_words = 0
            for sentence in self.content:
                nb_words += len(sentence)
            if nb_words < 101:
                print("WARNING : Text length is less than 100 words, some scores will be inaccurate.")
            print(nb_words)

        # Handling text that doesn't need to be converted
        elif any(isinstance(el, list) for el in content):
            self.content_type = "text"
            self.content = content

        # Handling text that was only converted into tokens
        elif isinstance(content, list):
            self.content_type = "text"
            content = ' '.join(content)
            self.content = [[token.text for token in sent] for sent in self.nlp(content).sents]
            #self.content = [[token.text for token in sent if not token.is_punct] for sent in self.nlp(content).sents]
            nb_words = 0
            for sentence in self.content:
                nb_words += len(sentence)
            if nb_words < 101:
                print("WARNING : Text length is less than 100 words, some scores will be inaccurate.")


        # Check if input is a corpus.
        else:
            # Reminder, structure needed is : dict => list of texts => list of sentences => list of words
            # TODO : check this with a bunch of edge cases
            if type(content) == dict:
                if isinstance(content[list(content.keys())[0]], list):
                    if isinstance(content[list(content.keys())[0]][0], list):
                        if isinstance(content[list(content.keys())[0]][0][0], list):
                            self.content_type = "corpus"
                            self.content = content
                            self.classes = list(content.keys())
            #else, check if the structure is dict[class][text].. and tokenize everything (warn user it'll take some time)
            #and then use that as the new structure

    def score(self, name):
        """
        This function calls the relevant score from the submodule common_scores, based on the information possessed by the current instance
        First, it determines whether we're dealing with a text or a corpus via the use of self.content_type
        Then, it checks whether the current text was compiled by checking if self.statistics or self.corpus_statistics exists.
        It then calls the score for a text, or returns a dictionary containing lists of scores in the same format as a corpus (dict{class[score]})

        :param name: Content of a text, or a corpus
        :type name: str, list(str), list(list(str)), converted into list(list(str)) or dict[class][text][sentence][token]
        """
        if name == "gfi":
            func = common_scores.GFI_score
        elif name == "ari":
            func = common_scores.ARI_score
        elif name == "fre":
            func = common_scores.FRE_score
        elif name == "fkgl":
            func = common_scores.FKGL_score
        elif name == "smog":
            func = common_scores.SMOG_score
        elif name == "rel":
            func = common_scores.REL_score

        if self.content_type == "text":
            if hasattr(self, "statistics"):
                return func(self.content, self.statistics)
            else:
                return func(self.content)
        elif self.content_type == "corpus":
            scores = {}
            for level in self.classes:
                scores[level] = []
                if hasattr(self, "corpus_statistics"):
                    for statistics in self.corpus_statistics[level]:
                        scores[level].append(func(None, statistics))
                else:
                    for text in self.content[level]:
                        scores[level].append(func(text))
            # Output part of the scores, first text for each class
            for level in self.classes:
                print("class", level, "text 0", "score" ,scores[level][0])
            return scores
        else:
            return -1
        
    def gfi(self):
        """Returns Gunning Fog Index"""
        return self.score("gfi")

    def ari(self):
        """Returns Automated Readability Index"""
        return self.score("ari")

    def fre(self):
        """Returns Flesch Reading Ease"""
        return self.score("fre")

    def fkgl(self):
        """Returns Flesch–Kincaid Grade Level"""
        return self.score("fkgl")

    def smog(self):
        """Returns Simple Measure of Gobbledygook"""
        return self.score("smog")

    def rel(self):
        """Returns Reading Ease Level (Adaptation of FRE for french)"""
        return self.score("rel")

    def scores(self):
        """
        Depending on type of content provided, returns a list of common readability scores (type=text),
        or returns a matrix containing the mean values for these scores, depending on the classes of the corpus (type=corpus) 

        :return: a pandas dataframe (or a list of scores)
        :rtype: Union[pandas.core.frame.DataFrame,list] 
        """
        #NOTE : Would be better to have this point to a scores_text() and scores_corpus(), which returns only one type.
        if self.content_type == "corpus":
            if hasattr(self, "corpus_statistics"):
                return common_scores.traditional_scores(self.content, self.corpus_statistics)
            else:
                return common_scores.traditional_scores(self.content)

        elif self.content_type == "text":
            scores = ["gfi","ari","fre","fkgl","smog","rel"]
            values = {}
            for score in scores:
                values[score] = self.score(score)
            return values

    def diversity(self, type, mode):
        if type == "ttr":
            func = diversity.type_token_ratio
        elif type == "ntr":
            func = diversity.noun_token_ratio

        if self.content_type == "text":
            return func(self.content, self.nlp, mode)
        elif self.content_type == "corpus":
            scores = {}
            for level in self.classes:
                scores[level] = []
                for text in self.content[level]:
                    scores[level].append(func(text, self.nlp, mode))
            # Output part of the scores, first text for each class
            for level in self.classes:
                print("class", level, "text 0", "score" ,scores[level][0])
            return scores
        else:
            return -1

    def perplexity(self):
        """
        Outputs pseudo-perplexity, which is derived from pseudo-log-likelihood scores.
        Please refer to this paper for more details : https://doi.org/10.18653%252Fv1%252F2020.acl-main.240

        :return: The pseudo-perplexity measure for a text, or for each text in a corpus.
        :rtype: float or list(float)
        """
        if not hasattr(self.perplexity_calculator, "model_loaded"):
            self.perplexity_calculator.load_model()
            print("Model is now loaded")
        print("Please be patient, pseudo-perplexity takes a lot of time to calculate.")
        if self.content_type == "text":
            return self.perplexity_calculator.PPPL_score_text(self.content)
        elif self.content_type == "corpus":
            return self.perplexity_calculator.PPPL_score(self.content)
        return -1
    
    def remove_outliers(self,perplex_scores,stddev_ratio):
        """
        Outputs a corpus, after removing texts which are considered to be "outliers", based on a standard deviation ratio
        A text is an outlier if its pseudo-perplexity value is lower or higher than this : mean +- standard_deviation * ratio
        In order to exploit this new corpus, you'll need to make a new Readability instance.
        For instance : new_r = Readability(r.remove_outliers(r.perplexity(),1))

        :return: a corpus, in a specific format where texts are represented as lists of sentences, which are lists of words.
        :rtype: dict[class][text][sentence][token]
        """
        if not hasattr(self.perplexity_calculator, "model_loaded"):
            self.perplexity_calculator.load_model()
            print("Model is now loaded")
        if self.content_type == "text":
            raise TypeError('Content type is not corpus, please load something else to use this function.')
        elif self.content_type == "corpus":
            return self.perplexity_calculator.remove_outliers(self.content,perplex_scores,stddev_ratio)
        return -1

    def compile(self):
        """
        Calculates a bunch of statistics to make some underlying functions faster.
        Returns a copy of a Readability instance, supplemented with a "statistics" or "corpus_statistics" attribute that can be used for other functions.
        """
        #TODO : debloat this and/or refactor it since we copy-paste almost the same below
        if self.content_type == "text":
            #I can probably do self.statistics.totalWords = 0 directly..
            totalWords = 0
            totalLongWords = 0
            totalSentences = len(self.content)
            totalCharacters = 0
            totalSyllables = 0
            nbPolysyllables = 0
            for sentence in self.content:
                totalWords += len(sentence)
                totalLongWords += len([token for token in sentence if len(token)>6])
                totalCharacters += sum(len(token) for token in sentence)
                totalSyllables += sum(utils.syllablesplit(word) for word in sentence)
                nbPolysyllables += sum(utils.syllablesplit(word) for word in sentence if utils.syllablesplit(word)>=3)
                #nbPolysyllables += sum(1 for word in sentence if utils.syllablesplit(word)>=3)
            self.statistics = Statistics()
            self.statistics.totalWords = totalWords
            self.statistics.totalLongWords = totalLongWords
            self.statistics.totalSentences = totalSentences
            self.statistics.totalCharacters = totalCharacters
            self.statistics.totalSyllables = totalSyllables
            self.statistics.nbPolysyllables = nbPolysyllables
            return self
        elif self.content_type == "corpus":
            stats = {}
            for level in self.classes:
                stats[level] = []
                for text in self.content[level]:
                    #This could be turned into a subroutine instead of copy pasting...
                    totalWords = 0
                    totalLongWords = 0
                    totalSentences = len(text)
                    totalCharacters = 0
                    totalSyllables = 0
                    nbPolysyllables = 0
                    for sentence in text:
                        totalWords += len(sentence)
                        totalLongWords += len([token for token in sentence if len(token)>6])
                        totalCharacters += sum(len(token) for token in sentence)
                        totalSyllables += sum(utils.syllablesplit(word) for word in sentence)
                        nbPolysyllables += sum(utils.syllablesplit(word) for word in sentence if utils.syllablesplit(word)>=3)
                        #nbPolysyllables += sum(1 for word in sentence if utils.syllablesplit(word)>=3)
                    statistics = Statistics()
                    #TODO : make code less bloated by doing something like this : 
                    #for p in params:
                    #    setattr(self.statistics, p, p[value])
                    statistics.totalWords = totalWords
                    statistics.totalLongWords = totalLongWords
                    statistics.totalSentences = totalSentences
                    statistics.totalCharacters = totalCharacters
                    statistics.totalSyllables = totalSyllables
                    statistics.nbPolysyllables = nbPolysyllables
                    stats[level].append(statistics)
            self.corpus_statistics = stats
            return self
        return -1

    def stats(self):
        """
        Prints to the console the contents of the statistics obtained for a text, or part of the statistics for a corpus.
        In this case, this will output the statistics for the first text for every class
        """
        if hasattr(self, "statistics"):
            for stat in self.statistics.__dict__:
                print(stat, "=", self.statistics.__dict__[stat])

        #maybe i should just return the mean values?
        elif hasattr(self, "corpus_statistics"):
            for level in self.classes:
                print("Class", level, "first text's values")
                for stat in self.corpus_statistics[level][0].__dict__:
                    print(stat, "=", self.corpus_statistics[level][0].__dict__[stat])
        else:
            print("You need to apply the .compile() function before in order to view this",self.content_type,"' statistics")


    # NOTE: Maybe this should go in the stats subfolder to have less bloat.
    def corpus_info(self):
        """
        Output several basic statistics such as number of texts, sentences, or tokens, alongside size of the vocabulary.
            
        :param corpus: Dictionary of lists of sentences (represented as a list of tokens)
        :type corpus: dict[class][text][sentence][token]

        :return: a pandas dataframe 
        :rtype: pandas.core.frame.DataFrame
        """
        if self.content_type != "corpus":
            raise TypeError("Current type is not recognized as corpus. Please provide a dictionary with the following format : dict[class][text][sentence][token]")
        else:
            # Extract the classes from the dictionary's keys.
            corpus = self.content
            levels = self.classes
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
                    temp_nbr_ph = min(len(text),1) # NOTE : not sure this is a good idea.
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
