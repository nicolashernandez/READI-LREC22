"""
The ParsedCollection module is a convenient way to group ParsedTexts together and enable functions that require group(s) of text by the processor.

Tentative structure is to make it a dict{list(ParsedText)} but for now let's just make it a list of texts and see what we can do.
"""
import copy
import math

import pandas as pd
import spacy
from scipy.stats import pearsonr
from ..utils import utils

class ParsedCollection:
    """
    The ParsedCollection class serves as a convenient way to group ParsedText together and access relevant ReadabilityProcessor functions.

    Might be created as a result of ReadabilityProcessor.parseCollection(), or might be instantiated on its own. Not sure for now.
    """
    def __init__(self, text_collection, readability_processor):
        """
        Constructor of the ParsedCollection class, creates the following attributes:
        self.readability_processor, self.content, self.scores, self.statistics
        However, none of the scores in self.scores will be initialized.
        We're supposing that texts are already in ParsedText(format).
        """
        self.readability_processor = readability_processor
        self.content = text_collection
        self.corpus_name = "TO_BE_IMPLEMENTED"

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
                for stat in list(self.statistics[label].keys()):
                    self.statistics[label][stat] += text.statistics[stat]
            self.statistics[label]["vocabulary"] = set()
            for text in self.content[label]:
                self.statistics[label]["vocabulary"] = self.statistics[label]["vocabulary"].union(text.statistics["vocabulary"])
            self.statistics[label]["totalTexts"] = len(self.content[label])
            self.statistics[label]["meanSentences"] = round(self.statistics[label]["totalSentences"] / len(self.content[label]),1)
            self.statistics[label]["meanTokens"] = round(self.statistics[label]["totalWords"] / len(self.content[label]),1)

    
    def show_statistics(self):
        """
        Prints to the console the contents of the statistics obtained for a text, or part of the statistics for a corpus.
        In this case, this will output the mean values of each score for each class.
        """
        for label in list(self.content.keys()):
            print(label + "------------------")
            for stat in list(self.statistics[label].keys()):
                if stat == "vocabulary":
                    print(stat, "=", len(self.statistics[label][stat]), "words")
                else:
                    print(stat, "=", self.statistics[label][stat])

    def remove_outliers(self,score_type = None, stddevratio=1):
        """
        Outputs a corpus, after removing texts which are considered to be "outliers", based on a standard deviation ratio.
        A text is an outlier if its value is lower or higher than this : mean +- standard_deviation * ratio
        In order to exploit this new corpus, you'll need to make a new Readability instance.
        For instance : new_r = Readability(r.remove_outliers(r.perplexity(),1))

        :return: a corpus, in a specific format where texts are represented as lists of sentences, which are lists of words.
        :rtype: dict[str][str][str][str]
        """
        #I can get the moy no problem, but I do need the stddev. eh
        texts = self.content
        moy = list(self.call_score(score_type).values())

        stddev = list()
        for index, label in enumerate(list(self.content.keys())):
            inner_stddev=0
            for text in texts[label]:
                inner_stddev += ((text.call_score(score_type)-moy[index])**2)/len(texts[label])
            inner_stddev = math.sqrt(inner_stddev)
            stddev.append(inner_stddev)

        outliers_indices = texts.copy()
        for index, label in enumerate(list(self.content.keys())):
            outliers_indices[label] = [idx for idx in range(len(texts[label])) if texts[label][idx].call_score(score_type) > moy[index] + (stddevratio * stddev[index]) or texts[label][idx].call_score(score_type) < moy[index] - (stddevratio * stddev[index])]
            print("nb textes enleves(",label,") :", len(outliers_indices[label]),sep="")
            print(outliers_indices[label])

        corpus_no_outliers = copy.deepcopy(texts)
        for label in list(self.content.keys()):
            offset = 0
            for index in outliers_indices[label][:]:
                corpus_no_outliers[label].pop(index - offset)
                offset += 1
            print("New number of texts for class", label, ":", len(corpus_no_outliers[label]))
        print("In order to use this new corpus, you'll have to make a new Readability instance.")

        #So.. we have a dictionary associating labels and ParsedTexts. need to make an import function in Readability I suppose.
        # Or do that now.
        #Some of the values have to be re-calculated though..
        #Not the ones within the ParsedTexts themselves, but the ones in ParsedCollection..
        #Or I can just re-init a new CorpusCollection instance. yeah. it does the init stuff, values of texts are known, but corpus specific values aren't.
        return ParsedCollection(corpus_no_outliers,self.readability_processor)

    def call_score(self,score_name, arguments=None, force=False):
        """
        Helper function that gets a score if it already exists, otherwise checks if it's available, if so call the relevant function from the ReadabilityProcessor
        Use of function is : instance.call_score("score_name", arguments:[arg1,arg2,argi..], force:bool)
        If the underlying function needs no additional arguments, just pass en empty list, e.g : instance.call_score("pppl",[],True)

        :param str score_name: Name of a score recognized by ReadabilityProcessor.informations.
        :param list(any) arguments: Values used to change behavior of underlying functions.
        :param bool force: Indicates whether to force the calculation of a score or not.
        """
        moy_score = dict()
        # Check if measure already calculated
        for label in list(self.content.keys()):
            # If so, then just get it
            if self.scores[score_name][label] is not None and not force:
                moy_score[label] = self.scores[score_name][label]
            elif self.readability_processor.check_score_and_dependencies_available(score_name):
                #Otherwise, get it if ParsedText items already calculated it, or get them to calculate it.
                moy = 0
                for text in self.content[label]:
                    moy += text.call_score(score_name, arguments, force)
                self.scores[score_name][label] = moy / len(self.content[label])
                moy_score[label] = self.scores[score_name][label]
            else:
                moy_score[label] = None
        return moy_score

    def show_scores(self,force=False,correlation=None):
        if force:
            for score_name in list(self.scores.keys()):
                self.scores[score_name] = self.call_score(score_name, force=True)
        df = []
        score_names = []
        if correlation == "pearson":
            pearson = []
            for score_name in list(self.scores.keys()):
                df.append(list(self.scores[score_name].values()))
                score_names.append(score_name)
                labels = []
                score_as_list = []
                if list(self.scores[score_name].values())[0] is None:
                    pearson.append(None)
                else:
                    for label in list(self.content.keys()):
                        for text in self.content[label]:
                            score_as_list.append(text.call_score(score_name))
                            labels.append(list(self.content.keys()).index(label))
                    pearson.append(pearsonr(score_as_list,labels)[0])
            df = pd.DataFrame(df,columns = list(self.content.keys()))
            df["Pearson Score"] = pearson
            df.index = score_names
            return df
        elif correlation is None:
            for score_name in list(self.scores.keys()):
                df.append(list(self.scores[score_name].values()))
                score_names.append(score_name)
        df = pd.DataFrame(df,columns = list(self.content.keys()))
        df.index = score_names
        return df
    

    # Traditional measures :
    def traditional_score(self,score_name):
        # NOTE: don't use self.call_score for this one, the reference to ParsedText.call_score prevents using ParsedText.statistics as a speed-up.
        # Since the functions share the same name, there probably exists a way to "access" the .traditional_score() method of ParsedText instead of
        # knowning how to get ReadabilityProcessor.score(), which is less useful.
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
    def perplexity(self,force=False):
        return self.call_score("pppl",[], force)
    
    def stub_rsrs(self,force=False):
        return self.call_score("rsrs",[], force)


    # Measures related to text diversity
    def diversity(self, ratio_type, formula_type=None, force=False):
        return self.call_score(ratio_type,[formula_type],force)

    def ttr(self, formula_type=None, force=False):
        """Returns Text Token Ratio: number of unique words / number of words"""
        return self.diversity("ttr", formula_type, force)

    def ntr(self, formula_type=None, force=False):
        """Returns Noun Token Ratio: number of nouns / number of nouns"""
        return self.diversity("ntr", formula_type, force)
    

    # Measures based on pre-existing word lists
    def dubois_proportion(self,filter_type="total",filter_value=None, force=False):
        return self.call_score("dubois_buyse_ratio",[filter_type, filter_value],force)

    def average_levenshtein_distance(self,mode="old20", force=False):
        return self.call_score(mode,[],force)
    def old20(self, formula_type=None, force=False):
        """Returns average Orthographic Levenshtein Distance 20 (OLD20) in a text"""
        return self.average_levenshtein_distance("old20", force)

    def pld20(self, formula_type=None, force=False):
        """Returns average Phonemic Levenshtein Distance 20 (OLD20)"""
        return self.average_levenshtein_distance("pld20", force)
    
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
        return self.call_score("cosine_similarity_tfidf",[mode],force)

    # NOTE: this seems to output the same values, whether we use text or lemmas, probably due to the type of model used.
    def lexical_cohesion_LDA(self ,mode="text", force=False):
        return self.call_score("cosine_similarity_LDA",[mode],force)

    # Following functions are meant to be used with a corpus/collection of texts, while they could output a score such as mean accuracy,
    # That would conflict with the scores that can be obtained with a text.. unless we do a corpus.informations attribute.. hmmm..
    def classify_corpus_SVM(self):
        return self.readability_processor.classify_corpus_SVM(self)
    def classify_corpus_MLP(self):
        return self.readability_processor.classify_corpus_MLP(self)
    def compare_ML_models(self):
        return self.readability_processor.compare_ML_models(self)

