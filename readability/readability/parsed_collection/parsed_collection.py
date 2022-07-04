"""
The ParsedCollection module is a convenient way to group ParsedTexts together and enable functions that require group(s) of text by the processor.

Tentative structure is to make it a dict{list(ParsedText)} but for now let's just make it a list of texts and see what we can do.
"""
import copy
import math

import pandas as pd
import spacy
from ..utils import utils

class ParsedCollection:
    """
    The ParsedCollection class serves as a convenient way to group ParsedText together and access relevant ReadabilityProcessor functions.

    Might be created as a result of ReadabilityProcessor.parseCollection(), or might be instantiated on its own. Not sure for now.
    """
    def __init__(self, list_of_texts, readability_processor):
        """
        Constructor of the ParsedCollection class, creates the following attributes:
        self.readability_processor, self.content, self.scores, self.statistics
        However, none of the scores in self.scores will be initialized.
        We're supposing that texts are already in ParsedText(format).
        """
        self.readability_processor = readability_processor

        # TODO: Placeholder creation of self.content, create a function that converts different types of (labels, texts) collections into the same structure
        # Also, if using a list of texts, make a dict(default = list_of_texts) for compatibility, we'll suppose every text has the same label.
        if isinstance(list_of_texts, str):
            raise TypeError("Input type is 'str', please provide a dict() or a list(list()) instead")
        # Recognising a list of lists, weird but ok.
        if any(isinstance(el, list) for el in list_of_texts):
            print("should be list of lists, can do something to convert there")
            print("just assign a label 1,2,3,4,5 to each label or something, doesn't really matter, but warn user.")
        self.content = dict(default = list_of_texts)

        # Initialize scores by setting them all to None
        self.scores = dict()
        for info in list(readability_processor.informations.keys()):
            self.scores[info] = dict()
            for label in list(self.content.keys()):
                self.scores[info][label] = None
        for info in list(readability_processor.excluded_informations.keys()):
            self.scores[info] = dict()
            for label in list(self.content.keys()):
                self.scores[info][label] = None
        

        self.statistics = dict()
        for label in list(self.content.keys()):
            self.statistics[label] = dict()
            self.statistics[label]["totalWords"] = 0
            self.statistics[label]["totalLongWords"] = 0
            self.statistics[label]["totalSentences"] = 0
            self.statistics[label]["totalCharacters"] = 0
            self.statistics[label]["totalSyllables"] = 0
            self.statistics[label]["nbPolysyllables"] = 0
            for text in self.content[label]:
                self.statistics[label]["totalSentences"] += len(text.content)
                for sentence in text.content:
                    self.statistics[label]["totalWords"] += len(sentence)
                    self.statistics[label]["totalLongWords"] += len([token for token in sentence if len(token)>6])
                    self.statistics[label]["totalCharacters"] += sum(len(token) for token in sentence)
                    self.statistics[label]["totalSyllables"] += sum(utils.syllablesplit(word) for word in sentence)
                    self.statistics[label]["nbPolysyllables"] += sum(utils.syllablesplit(word) for word in sentence if utils.syllablesplit(word)>=3)
                    #self.statistics["nbPolysyllables"] += sum(1 for word in sentence if utils.syllablesplit(word)>=3)
    
    def show_statistics(self):
        """
        Prints to the console the contents of the statistics obtained for a text, or part of the statistics for a corpus.
        In this case, this will output the mean values of each score for each class.
        """
        for label in list(self.content.keys()):
            for stat in list(self.statistics[label].keys()):
                print(stat, "=", self.statistics[label][stat])
        #TODO : re-use corpus.info() from utils/corpus_utils to show more stuff

    def call_score(self,score_name):
        moy_score = dict()
        # Check if measure already calculated
        for label in list(self.content.keys()):
            # If so, then just get it
            if self.scores[score_name][label] is not None:
                moy_score[label] = self.scores[score_name][label]
            elif self.readability_processor.check_score_and_dependencies_available(score_name):
                #Otherwise, get it if ParsedText items already calculated it, or get them to calculate it.
                moy = 0
                for text in self.content[label]:
                    moy += text.call_score(score_name)
                self.scores[score_name][label] = moy / len(self.content[label])
                moy_score[label] = self.scores[score_name][label]
            else:
                moy_score[label] = None
        return moy_score

    def show_scores(self,force=False):
        df = []
        if force:
            for score_name in list(self.scores.keys()):
                self.scores[score_name] = self.call_score(score_name)
        
        # Append each already-calculated score to a dataframe
        df.append(self.scores)
        df = pd.DataFrame(df)
        return df
    

    # Traditional measures :
    def traditional_score(self,score_name):
        moy_score = dict()
        # Check if scores exist, otherwise calculate 
        for label in list(self.content.keys()):
            if self.scores[score_name][label] == None:
                # Get every ParsedText score (or let them calculate it)
                moy = 0
                for text in self.content[label]:
                    moy += text.traditional_score(score_name)
                self.scores[score_name][label] = moy / len(self.content[label])
                moy_score[label] = self.scores[score_name][label]
            else:
                # Just get the score
                moy_score[label] = self.scores[score_name][label]
        # TODO : Add pearson coefficients too. According to stats/common_scores, need to flatten each text into a list, and have a list of labels with corresponding indexes.
        # Should be quick to reproduce
        return moy_score
    
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
        moy_score = dict()
        for label in list(self.content.keys()):
            if self.scores["pppl"][label] == None:
                print("Now calculating perplexity for class:",label)
                moy = 0
                for text in self.content[label]:
                    moy += text.perplexity()
                self.scores["pppl"][label] = moy / len(self.content[label])
                moy_score[label] = self.scores["pppl"][label]
            else:
                moy_score[label] = self.scores["pppl"][label]
        return moy_score
    
    def stub_rsrs(self):
        moy_score = dict()
        for label in list(self.content.keys()):
            if self.scores["rsrs"][label] == None:
                moy = 0
                for text in self.content[label]:
                    moy += text.stub_rsrs()
                self.scores["rsrs"][label] = moy / len(self.content[label])
                moy_score[label] = self.scores["rsrs"][label]
            else:
                moy_score[label] = self.scores["rsrs"][label]
        return moy_score


    # Measures related to text diversity
    def diversity(self, ratio_type, formula_type=None, force=False):
        moy_score = dict()
        for label in list(self.content.keys()):
            if self.scores[ratio_type][label] == None or force:
                moy = 0
                for text in self.content[label]:
                    moy += text.diversity(ratio_type,formula_type,force)
                self.scores[ratio_type][label] = moy / len(self.content[label])
                moy_score[label] = self.scores[ratio_type][label]
            else:
                moy_score[label] = self.scores[ratio_type][label]
        return moy_score

    def ttr(self, formula_type=None, force=False):
        """Returns Text Token Ratio: number of unique words / number of words"""
        return self.diversity("ttr", formula_type, force)

    def ntr(self, formula_type=None, force=False):
        """Returns Noun Token Ratio: number of nouns / number of nouns"""
        return self.diversity("ntr", formula_type, force)
    

    # Measures based on pre-existing word lists
    def dubois_proportion(self,filter_type="total",filter_value=None, force=False):
        moy_score = dict()
        for label in list(self.content.keys()):
            if self.scores["dubois_buyse_ratio"][label] == None or force:
                moy = 0
                for text in self.content[label]:
                    moy += text.dubois_proportion(filter_type,filter_value,force)
                self.scores["dubois_buyse_ratio"][label] = moy / len(self.content[label])
                moy_score[label] = self.scores["dubois_buyse_ratio"][label]
            else:
                moy_score[label] = self.scores["dubois_buyse_ratio"][label]
        return moy_score

    def average_levenshtein_distance(self,mode="old20", force=False):
        moy_score = dict()
        for label in list(self.content.keys()):
            if self.scores[mode][label] == None or force:
                moy = 0
                for text in self.content[label]:
                    moy += text.average_levenshtein_distance(mode,force)
                self.scores[mode][label] = moy / len(self.content[label])
                moy_score[label] = self.scores[mode][label]
            else:
                moy_score[label] = self.scores[mode][label]
        return moy_score

    # NOTE : might do these 3 at start-up instead.
    #def count_pronouns(self,mode="text"):
    #    if "nb_pronouns" in list(self.statistics.keys()):
    #        if self.statistics["nb_pronouns"] == None:
    #            self.statistics["nb_pronouns"] = self.readability_processor.count_pronouns(self.content,mode)
    #    else: 
    #        self.statistics["nb_pronouns"] = self.readability_processor.count_pronouns(self.content,mode)
    #    return self.statistics["nb_pronouns"]
    
    #def count_articles(self,mode="text"):
    #    if "nb_articles" in list(self.statistics.keys()):
    #        if self.statistics["nb_articles"] == None:
    #            self.statistics["nb_articles"] = self.readability_processor.count_articles(self.content,mode)
    #    else: 
    #        self.statistics["nb_articles"] = self.readability_processor.count_articles(self.content,mode)
    #    return self.statistics["nb_articles"]
        
    #def count_proper_nouns(self,mode="text"):
    #    if "nb_proper_nouns" in list(self.statistics.keys()):
    #        if self.statistics["nb_proper_nouns"] == None:
    #            self.statistics["nb_proper_nouns"] = self.readability_processor.count_proper_nouns(self.content,mode)
    #    else: 
    #        self.statistics["nb_proper_nouns"] = self.readability_processor.count_proper_nouns(self.content,mode)
    #    return self.statistics["nb_proper_nouns"]

    def lexical_cohesion_tfidf(self, mode="text", force=False):
        moy_score = dict()
        for label in list(self.content.keys()):
            if self.scores["cosine_similarity_tfidf"][label] == None or force:
                moy = 0
                for text in self.content[label]:
                    moy += text.lexical_cohesion_tfidf(mode, force)
                self.scores["cosine_similarity_tfidf"][label] = moy / len(self.content[label])
                moy_score[label] = self.scores["cosine_similarity_tfidf"][label]
            else:
                moy_score[label] = self.scores["cosine_similarity_tfidf"][label]
        return moy_score

    # NOTE: this seems to output the same values, whether we use text or lemmas, probably due to the type of model used.
    def lexical_cohesion_LDA(self, mode="text", force=False):
        moy_score = dict()
        for label in list(self.content.keys()):
            if self.scores["cosine_similarity_LDA"][label] == None or force:
                moy = 0
                for text in self.content[label]:
                    moy += text.lexical_cohesion_LDA(mode, force)
                self.scores["cosine_similarity_LDA"][label] = moy / len(self.content[label])
                moy_score[label] = self.scores["cosine_similarity_LDA"][label]
            else:
                moy_score[label] = self.scores["cosine_similarity_LDA"][label]
        return moy_score