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
        self.content = content
        self.readability_processor = readability_processor

        # Converting text into a list(list(str)) format in order to properly seperate sentences and tokens.
        self.content = utils.convert_text_to_sentences(content,self.nlp)

        self.scores = dict() # Scores don't get initialized since
        self.statistics = dict()
        #TODO : put what's in readability.compile() here.
    
    def show_text(self):
        return utils.convert_text_to_string(self.content)

    def show_scores(self,force=False):
        # TODO : Create a dataframe, append each already-calculated score
        # Then if force=True => for every non-calculated score =>
        #   Check if score appears in .readability_processor.methods:
        #       Calculate that score and append to dataframe
        # Otherwise, append each score but add NaN or NA or something similar
        # Then for every other score in .readability_processor.excluded_methods:
        #   Append each of these scores but with Nan or NA
        # No need to store the dataframe since checking if scores appear in dict should take a miniscule amount of time
        return -1

    def show_statistics(self):
        # TODO: for each stat in .statistics, append that to a dataframe and print it.
        return -1


    # NOTE : Explicitely naming each of the functions but probably exists a better way to set certain function names from the ReadabilityProcessor instance
    # Exemple, perplexity :
    def perplexity(self):
        if self.scores["pppl"] == None:
            self.scores["pppl"] = self.readability_processor.perplexity()
        return self.scores["pppl"]

    # Should work even with functions that need arguments :
    # Reminder that type is 'ttr' or 'ntr', mode is 'corrected' or 'root' or defaulting to 'normal'
    # Probably a better idea to just make a ttr() and ntr() function instead to avoid confusion down the line.
    def diversity(self, type, mode=None):
        if self.scores["pppl"] == None:
            self.scores["pppl"] = self.readability_processor.perplexity(type, mode)
        return self.scores["pppl"]

    # I probably could make a call_function(func_args) subroutine, but what happens if func_args is empty, does it add nothing or does it break.