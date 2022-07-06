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
        self.statistics["vocabulary"] = set()
        for sentence in self.content:
            self.statistics["totalWords"] += len(sentence)
            self.statistics["totalLongWords"] += len([token for token in sentence if len(token)>6])
            self.statistics["totalCharacters"] += sum(len(token) for token in sentence)
            self.statistics["totalSyllables"] += sum(utils.syllablesplit(word) for word in sentence)
            self.statistics["nbPolysyllables"] += sum(utils.syllablesplit(word) for word in sentence if utils.syllablesplit(word)>=3)
            #self.statistics["nbPolysyllables"] += sum(1 for word in sentence if utils.syllablesplit(word)>=3)
            for token in sentence:
                self.statistics["vocabulary"].add(token)
            
    
    def show_text(self):
        return utils.convert_text_to_string(self.content)

    def show_statistics(self):
        """
        Prints to the console the contents of the statistics obtained for a text, or part of the statistics for a corpus.
        In this case, this will output the mean values of each score for each class.
        """
        for stat in list(self.statistics.keys()):
            print(stat, "=", self.statistics[stat])
        return None

    def call_score(self, score_name, arguments=None, force=False):
        """
        Helper function that gets a score if it already exists, otherwise checks if it's available, if so call the relevant function from the ReadabilityProcessor
        Use of function is : instance.call_score("score_name", arguments:[arg1,arg2,argi..], force:bool)
        If the underlying function needs no additional arguments, just pass en empty list, e.g : instance.call_score("pppl",[],True)

        :param str score_name: Name of a score recognized by ReadabilityProcessor.informations.
        :param list(any) arguments: Values used to change behavior of underlying functions.
        :param bool force: Indicates whether to force the calculation of a score or not.
        """
        # check if score_name already in scores:
        if self.scores[score_name] is not None and not force:
            return self.scores[score_name]
        # otherwise check if score_name is available in processor:
        elif self.readability_processor.check_score_and_dependencies_available(score_name):
            # If so, then call function based on informations
            func = self.readability_processor.informations[score_name]["function"]
            if arguments is None:
                arguments = self.readability_processor.informations[score_name]["default_arguments"].values()
                #print("WARNING: defaulting to default arguments :", arguments)

            self.scores[score_name] = func(self.content, *(arguments))
            return self.scores[score_name]
        # If function is unavailable, return None to indicate so.
        else:
            return None

    def show_scores(self,force=False):
        """
        Returns a dataframe containing each already calculated score, can also force calculation with default values.
        
        :param bool force: Indicates whether to force the calculation of each score
        """
        # NOTE: one behavior could be added : return every score if possible, and calculate the rest, instead of calculating everything.
        df = []
        if force:
            for score_name in list(self.scores.keys()):
                self.scores[score_name] = self.call_score(score_name,force=True)
        # Append each already-calculated score to a dataframe
        df.append(self.scores)
        df = pd.DataFrame(df)
        return df

    # Traditional measures
    
    def traditional_score(self,score_name,force=False):
        """
        Called by methods : gfi | ari | fre | fkgl | smog | rel. Serves as a entry-point to the underlying function "score" of ReadabilityProcessor
        
        :param str score_name: Name of a score recognized by ReadabilityProcessor.informations.
        :param bool force: Indicates whether to force the calculation of a score or not.
        """
        return self.call_score(score_name,[self.statistics],force)

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
    def perplexity(self, force=False):
        """
        Outputs pseudo-perplexity, which is derived from pseudo-log-likelihood scores.

        :param bool force: Indicates whether to force the calculation of a score or not.
        :return: The pseudo-perplexity measure for a text, or for each text in a corpus.
        :rtype: float
        """
        return(self.call_score("pppl",[],force))
    
    def stub_rsrs(self, force=False):
        return(self.call_score("rsrs",[],force))


    # Measures related to text diversity
    def diversity(self, ratio_type, formula_type=None, force=False):
        """
        Outputs a measure of text diversity based on which feature to use, and which version of the formula is used.
        Default formula is 'nb lexical items' / 'nb unique lexical items',
        'root' formula uses the square root for the denominator,
        'corrected' formula mutliplies the number of words by two before taking the square root for the denominator.

        :param str ratio_type: Which text diversity measure to use: "ttr" is text token ratio, "ntr" is noun token ratio
        :param str formula_type: What kind of formula to use: "corrected", "root", and default standard are available for token ratios.
        :param bool force: Indicates whether to force the calculation of a score or not.
        """
        return self.call_score(ratio_type,[formula_type],force)

    def ttr(self, formula_type=None, force=False):
        """Returns Text Token Ratio: number of unique words / number of words"""
        return self.diversity("ttr", formula_type, force)

    def ntr(self, formula_type=None, force=False):
        """Returns Noun Token Ratio: number of nouns / number of nouns"""
        return self.diversity("ntr", formula_type, force)
    

    # Measures based on pre-existing word lists
    def dubois_proportion(self,filter_type="total", filter_value=None, force=False):
        """
        Outputs the proportion of words included in the Dubois-Buyse word list.
        Can specify the ratio for words appearing in specific echelons, ages or three-year cycles.

        :param str filter_type: Which variable to use to filter the word list : 'echelon', 'age', or 'cycle'
        :param str filter_value: Value (or iterable containing two values) for subsetting the word list.
        :type filter_value: Union[int, tuple, list]
        :param bool force: Indicates whether to force the calculation of a score or not.
        :return: a ratio of words in the current text, that appear in the Dubois-Buyse word list.
        :rtype: float
        """
        return self.call_score("dubois_buyse_ratio",[filter_type,filter_value],force)

    def average_levenshtein_distance(self, mode="old20", force=False):
        """
        Returns the average Orthographic Levenshtein Distance 20 (OLD20), or its phonemic equivalent (PLD20).
        Currently using the Lexique 3.0 database for French texts, version 3.83. More details here : http://www.lexique.org/
        OLD20 is an alternative to the orthographical neighbourhood index that has been shown to correlate with text difficulty.

        :param str mode: What kind of value to return, OLD20 or PLD20.
        :param bool force: Indicates whether to force the calculation of a score or not.
        :return: Average of OLD20 or PLD20 for each word in current text
        :rtype: Union[float,dict[str][list(float)]]
        """
        return self.call_score(mode,[],force)

    def old20(self, formula_type=None, force=False):
        """Returns average Orthographic Levenshtein Distance 20 (OLD20) in a text"""
        return self.average_levenshtein_distance("old20", force)

    def pld20(self, formula_type=None, force=False):
        """Returns average Phonemic Levenshtein Distance 20 (OLD20)"""
        return self.average_levenshtein_distance("pld20", force)


    # Measures based on text cohesion
    # NOTE : might do the following 3 at start-up instead.
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
        return self.call_score("cosine_similarity_tfidf",[mode],force)

    # NOTE: this seems to output the same values, whether we use text or lemmas, probably due to the type of model used.
    def lexical_cohesion_LDA(self ,mode="text", force=False):
        return self.call_score("cosine_similarity_LDA",[mode],force)

    # NOTE: 
