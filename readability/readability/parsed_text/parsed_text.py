"""
The ParsedText class serves as an interface between a text and a readability_processor instance in order to store useful readability measures.

It is meant to be created as a result of readability_processor.parse() since it uses the processor in order to know which measures are available, and
have access to the resources necessary to calculate them.
"""
import copy
import math

import pandas as pd
import spacy
from ..utils import utils

# NOTE: There probably exists a better way to create an Statistics object as an attribute of ParsedText
class Statistics:
    pass

class ParsedText:
    """
    The ParsedText class serves as an interface between a text and a ReadabilityProcessor instance in order to store useful readability measures.

    It is meant to be created as a result of ReadabilityProcessor.parse() since it uses the processor in order to know which measures are available, and
    have access to the resources necessary to calculate them.
    List of methods : __init__, show_text(), show_scores(), show_statistics().
    It also contains callers to ReadabilityProcessor methods that will be of the same name, please refer to its documentation to know which ones.
    List of attributes : content, readability_processor, statistics, scores
    """
    def __init__(self, content, readability_processor):
        """
        Constructor of the ParsedText class, won't return any value but creates the attributes :
        self.content, self.scores, self.statistics, self.readability_processor
        However, none of the scores in self.scores will be initialized.

        :param content: Content of a text.
        :type content: str, list(str), list(list(str)), converted into list(list(str))
        :param scores: Language the text was written in, in order to adapt some scores.
        :type scores: Probably a dict
        :param statistics: Common values used by various measures (Such as number of words, number of sentences, etc)
        :type statistics: Probably a dict
        :param str nlp: Type of NLP processor to use, indicated by a "type_subtype" string.
        :param readability_processor: Type of processor to use for the calculation of pseudo-perplexity
        :type readability_processor: ReadabilityProcessor
        """
        # NOTE: Maybe I should keep the str and list(str) variants of the text content stored in .statistics in order to re-use it later
        # Instead of potentially converting from sentences to text or from text to sentences several times.
        self.readability_processor = readability_processor

        # Converting text into a list(list(str)) format in order to properly seperate sentences and tokens.
        self.content = utils.convert_text_to_sentences(content,readability_processor.nlp)

        # Initialize scores by setting them all to None
        self.scores = dict()
        for info in list(readability_processor.informations.keys()):
            self.scores[info] = None
        for info in list(readability_processor.excluded_informations.keys()):
            self.scores[info] = None

        self.statistics = dict()
        self.statistics["totalWords"] = 0
        self.statistics["totalLongWords"] = 0
        self.statistics["totalSentences"] = len(self.content)
        self.statistics["totalCharacters"] = 0
        self.statistics["totalSyllables"] = 0
        self.statistics["nbPolysyllables"] = 0
        for sentence in self.content:
            self.statistics["totalWords"] += len(sentence)
            self.statistics["totalLongWords"] += len([token for token in sentence if len(token)>6])
            self.statistics["totalCharacters"] += sum(len(token) for token in sentence)
            self.statistics["totalSyllables"] += sum(utils.syllablesplit(word) for word in sentence)
            self.statistics["nbPolysyllables"] += sum(utils.syllablesplit(word) for word in sentence if utils.syllablesplit(word)>=3)
            #self.statistics["nbPolysyllables"] += sum(1 for word in sentence if utils.syllablesplit(word)>=3)
    
    def show_text(self):
        return utils.convert_text_to_string(self.content)

    def show_statistics(self):
        """
        Prints to the console the contents of the statistics obtained for a text, or part of the statistics for a corpus.
        In this case, this will output the mean values of each score for each class.
        """
        for stat in list(self.statistics.keys()):
            print(stat, "=", self.statistics[stat])
        return self.statistics

    def call_score(self,score_name):
        # check if score_name already in scores:
        if self.scores[score_name] is not None:
            return self.scores[score_name]
        # otherwise check if score_name is available in processor:
        elif self.readability_processor.check_score_and_dependencies_available(score_name):
            # If so, then call function based on informations
            func = self.readability_processor.informations[score_name]["function"]
            default_args = self.readability_processor.informations[score_name]["default_arguments"].values()
            self.scores[score_name] = func(self.content, *(default_args))
            return self.scores[score_name]
        # If function is unavailable, return None to indicate so.
        return None

    def show_scores(self,force=False):
        df = []
        if force:
            for score_name in list(self.scores.keys()):
                self.scores[score_name] = self.call_score(score_name)
        
        # Append each already-calculated score to a dataframe
        df.append(self.scores)
        df = pd.DataFrame(df)
        return df

    

    # Traditional measures
    def traditional_score(self,score_name):
        if self.scores[score_name] == None:
            self.scores[score_name] = self.readability_processor.score(score_name,self.content,self.statistics)
        return self.scores[score_name]
    
    def gfi(self):
        """Returns Gunning Fog Index"""
        return self.traditional_score("gfi")

    def ari(self):
        """Returns Automated Readability Index"""
        return self.traditional_score("ari")

    def fre(self):
        """Returns Flesch Reading Ease"""
        return self.traditional_score("fre")

    def fkgl(self):
        """Returns Flesch–Kincaid Grade Level"""
        return self.traditional_score("fkgl")

    def smog(self):
        """Returns Simple Measure of Gobbledygook"""
        return self.traditional_score("smog")

    def rel(self):
        """Returns Reading Ease Level (Adaptation of FRE for french)"""
        return self.traditional_score("rel")


    # Measures related to perplexity
    def perplexity(self):
        if self.scores["pppl"] == None:
            self.scores["pppl"] = self.readability_processor.perplexity(self.content)
        return self.scores["pppl"]
    
    def stub_rsrs(self):
        if self.scores["rsrs"] == None:
            self.scores["rsrs"] = self.readability_processor.stub_rsrs(self.content)
        return self.scores["rsrs"]


    # Measures related to text diversity
    def diversity(self, ratio_type, formula_type=None, force=False):
        if self.scores[ratio_type] == None or force:
            self.scores[ratio_type] = self.readability_processor.diversity(self.content,ratio_type,formula_type)
        return self.scores[ratio_type]

    def ttr(self, formula_type=None, force=False):
        """Returns Text Token Ratio: number of unique words / number of words"""
        return self.diversity("ttr", formula_type, force)

    def ntr(self, formula_type=None, force=False):
        """Returns Noun Token Ratio: number of nouns / number of nouns"""
        return self.diversity("ntr", formula_type, force)
    

    # Measures based on pre-existing word lists
    def dubois_proportion(self,filter_type="total",filter_value=None, force=False):
        if self.scores["dubois_buyse_ratio"] == None or force:
            self.scores["dubois_buyse_ratio"] = self.readability_processor.dubois_proportion(self.content,filter_type,filter_value)
        return self.scores["dubois_buyse_ratio"]

    def average_levenshtein_distance(self,mode="old20", force=False):
        if self.scores[mode] == None or force:
            self.scores[mode] = self.readability_processor.average_levenshtein_distance(self.content,mode)
        return self.scores[mode]

    # NOTE : might do these 3 at start-up instead.
    def count_pronouns(self,mode="text"):
        if "nb_pronouns" in list(self.statistics.keys()):
            if self.statistics["nb_pronouns"] == None:
                self.statistics["nb_pronouns"] = self.readability_processor.count_pronouns(self.content,mode)
        else: 
            self.statistics["nb_pronouns"] = self.readability_processor.count_pronouns(self.content,mode)
        return self.statistics["nb_pronouns"]
    
    def count_articles(self,mode="text"):
        if "nb_articles" in list(self.statistics.keys()):
            if self.statistics["nb_articles"] == None:
                self.statistics["nb_articles"] = self.readability_processor.count_articles(self.content,mode)
        else: 
            self.statistics["nb_articles"] = self.readability_processor.count_articles(self.content,mode)
        return self.statistics["nb_articles"]
        
    def count_proper_nouns(self,mode="text"):
        if "nb_proper_nouns" in list(self.statistics.keys()):
            if self.statistics["nb_proper_nouns"] == None:
                self.statistics["nb_proper_nouns"] = self.readability_processor.count_proper_nouns(self.content,mode)
        else: 
            self.statistics["nb_proper_nouns"] = self.readability_processor.count_proper_nouns(self.content,mode)
        return self.statistics["nb_proper_nouns"]

    def lexical_cohesion_tfidf(self, mode="text", force=False):
        if self.scores["cosine_similarity_tfidf"] == None or force:
            self.scores["cosine_similarity_tfidf"] = self.readability_processor.lexical_cohesion_tfidf(self.content,mode)
        return self.scores["cosine_similarity_tfidf"]

    # NOTE: this seems to output the same values, whether we use text or lemmas, probably due to the type of model used.
    def lexical_cohesion_LDA(self ,mode="text", force=False):
        if self.scores["cosine_similarity_LDA"] == None or force:
            self.scores["cosine_similarity_LDA"] = self.readability_processor.lexical_cohesion_LDA(self.content,mode)
        return self.scores["cosine_similarity_LDA"]
