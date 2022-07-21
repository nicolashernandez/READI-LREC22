"""
The ParsedText module contains the ParsedText class, serving as an interface between a text and a readability_processor instance.

It is used to store various measures and calculations that can be used across a variety of formulas or applications, in order to speed up the process.
Furthermore, it is used to store and access the services proposed by the library, such as text readability estimators, or testing machine learning models.
This class is meant to be created as a result of ReadabilityProcessor.parse() since it uses the processor in order to know which measures are available, and
have access to the external resources necessary to calculate them.
"""
import copy
import math

import pandas as pd
import spacy
from ..utils import utils

class ParsedText:
    """
    The ParsedText class serves as an interface between a text and a ReadabilityProcessor instance in order to store and output useful readability measures.

    It is meant to be created as a result of ReadabilityProcessor.parse() since it uses the processor in order to know which measures are available, and
    have access to the resources necessary to calculate them.
    List of methods : __init__, show_text(), call_score(), show_available_scores(), show_scores(), show_statistics().
    It also contains accessor functions based on ReadabilityProcessor methods, sharing the same name, these use the helper function call_score() in order to work.
    List of attributes : content, readability_processor, statistics, scores
    """
    def __init__(self, content, readability_processor):
        """
        Constructor of the ParsedText class, creates the 'content', 'scores', 'statistics', and 'readability_processor' attributes.

        Keep in mind that the scores default to None since they haven't been calculated yet.

        :param content: Content of a text.
        :type content: str, list(str), list(list(str)), converted into list(list(str))
        :param dict scores: Language the text was written in, in order to adapt some scores.
        :param dict statistics: Common values used by various measures (Such as number of words, number of sentences, etc)
        :param str nlp: Type of NLP processor to use, indicated by a "type_subtype" string.
        :param ReadabilityProcessor readability_processor: Type of processor to use for the calculation of pseudo-perplexity
        """
        self.readability_processor = readability_processor

        # Converting text into a list(list(str)) format in order to properly seperate sentences and tokens.
        self.content = utils.convert_text_to_sentences(content,readability_processor.nlp)

        # Initialize scores by setting them all to None
        self.scores = dict()
        for info in list(readability_processor.informations.keys()):
            self.scores[info] = None
        for info in list(readability_processor.excluded_informations.keys()):
            self.scores[info] = None

        # Calculate common statistics that can be used as part of more complex features.
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
        """Prints to the console the text itself."""
        return utils.convert_text_to_string(self.content)

    def show_statistics(self):
        """Prints to the console the contents of the statistics obtained for a text."""
        for stat in list(self.statistics.keys()):
            if stat == "vocabulary":
                print(stat, "=", len(self.statistics[stat]), "words")
            else:
                print(stat, "=", self.statistics[stat])
        return None

    def call_score(self, score_name, arguments=None, force=False):
        """
        Helper function that gets a score if it already exists, otherwise checks if it's available, if so call the relevant function from the ReadabilityProcessor
        
        Use of function is: instance.call_score(score_name:str, arguments:list(argi), force:bool)
        If the underlying function needs no additional arguments, just pass en empty list, e.g : instance.call_score("pppl",[],True)

        :param str score_name: Name of a score recognized by ReadabilityProcessor.informations.
        :param list(any) arguments: Values used to change behavior of underlying functions.
        :param bool force: Indicates whether to force the calculation of a score or not.
        """
        # Check if score_name already in scores:
        if self.scores[score_name] is not None and not force:
            return self.scores[score_name]
        # Otherwise check if score_name is available in processor:
        elif self.readability_processor.check_score_and_dependencies_available(score_name):
            # If so, then call function based on informations and provided arguments (if any)
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
        Returns a dataframe containing each calculated score, can force calculation with default values.
        
        :param bool force: Indicates whether to force the calculation of each score
        """
        # TODO: one behavior could be added : return every score if possible, and calculate the rest, instead of calculating everything.
        df = []
        if force:
            for score_name in list(self.scores.keys()):
                self.scores[score_name] = self.call_score(score_name,force=True)
        # Append each already-calculated score to a dataframe
        df.append(self.scores)
        df = pd.DataFrame(df)
        return df

    def show_available_scores(self):
        """Prints currently 'available' scores' names in a list"""
        return list(self.scores.keys())

    # Traditional measures
    def traditional_score(self,score_name,force=False):
        """
        Called by methods : gfi | ari | fre | fkgl | smog | rel. Serves as a entry-point to function "traditional_score" of ReadabilityProcessor.
        
        :param str score_name: Name of a score recognized by ReadabilityProcessor.informations.
        :param bool force: Indicates whether to force the calculation of a score or not.
        """
        return self.call_score(score_name,[self.statistics],force)

    def gfi(self):
        """
        Outputs the Gunning fog index, a 1952 readability test estimating the years of formal education needed to understand a text on the first reading.
        The scale goes from 6 to 18, starting at the sixth grade in the United States.
        The formula is : 0.4 * ( (words/sentences) + 100 * (complex words / words) )
        """
        return self.traditional_score("gfi")

    def ari(self):
        """
        Outputs the Automated readability index, a 1967 readability test estimating the US grade level needed to comprehend a text
        The scale goes from 1 to 14, corresponding to age 5 to 18.
        The formula is 4.71 * (characters / words) + 0.5 (words / sentences) - 21.43
        """
        return self.traditional_score("ari")

    def fre(self):
        """
        Outputs the Flesch reading ease, a 1975 readability test estimating the US school level needed to comprehend a text
        The scale goes from 100 to 0, corresponding to Grade 5 at score 100, up to post-college below score 30.
        The formula is 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
        """
        return self.traditional_score("fre")

    def fkgl(self):
        """
        Outputs the Flesch–Kincaid grade level, a 1975 readability test estimating the US grade level needed to comprehend a text
        The scale is meant to be a one to one representation, a score of 5 means that the text should be appropriate for fifth graders.
        The formula is 0.39 * (total words / total sentences)+11.8*(total syllables / total words) - 15.59
        """
        return self.traditional_score("fkgl")

    def smog(self):
        """
        Outputs the Simple Measure of Gobbledygook, a 1969 readability test estimating the years of education needed to understand a text
        The scale is meant to be a one to one representation, a score of 5 means that the text should be appropriate for fifth graders.
        The formula is 1.043 * Square root (Number of polysyllables * (30 / number of sentences)) + 3.1291
        """
        return self.traditional_score("smog")

    def rel(self):
        """
        Outputs the Reading Ease Level, an adaptation of Flesch's reading ease for the French language,
        with changes to the coefficients taking into account the difference in length between French and English words.
        The formula is 207 - 1.015 * (Number of words / Number of sentences) - 73.6 * (Number of syllables / Number of words)
        """
        return self.traditional_score("rel")


    # Measures related to perplexity
    def perplexity(self, force=False):
        """
        Outputs pseudo-perplexity, which is derived from pseudo-log-likelihood scores.
        Please refer to this paper for more details : https://doi.org/10.18653%252Fv1%252F2020.acl-main.240

        :param bool force: Indicates whether to force the calculation of a score or not.
        :return: The pseudo-perplexity measure for a text, or for each text in a corpus.
        :rtype: float
        """
        return(self.call_score("pppl",[],force))
    
    def stub_rsrs(self, force=False):
        """Not implemented yet, please check submodule stats/rsrs for implementation details."""
        #TODO : check submodule stats/rsrs for implementation details
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
    def dubois_buyse_ratio(self,filter_type="total", filter_value=None, force=False):
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

    def old20(self, force=False):
        """Returns average Orthographic Levenshtein Distance 20 (OLD20) in a text"""
        return self.average_levenshtein_distance("old20", force)

    def pld20(self, force=False):
        """Returns average Phonemic Levenshtein Distance 20 (OLD20)"""
        return self.average_levenshtein_distance("pld20", force)


    # Measures based on text cohesion
    # NOTE : might do the following 3 at start-up instead.
    def count_pronouns(self,mode="text"):
        """Returns number of pronouns in a text"""
        if "nb_pronouns" in list(self.statistics.keys()):
            if self.statistics["nb_pronouns"] == None:
                self.statistics["nb_pronouns"] = self.readability_processor.count_pronouns(self.content,mode)
        else: 
            self.statistics["nb_pronouns"] = self.readability_processor.count_pronouns(self.content,mode)
        return self.statistics["nb_pronouns"]
    
    def count_articles(self,mode="text"):
        """Returns number of articles in a text"""
        if "nb_articles" in list(self.statistics.keys()):
            if self.statistics["nb_articles"] == None:
                self.statistics["nb_articles"] = self.readability_processor.count_articles(self.content,mode)
        else: 
            self.statistics["nb_articles"] = self.readability_processor.count_articles(self.content,mode)
        return self.statistics["nb_articles"]
        
    def count_proper_nouns(self,mode="text"):
        """Returns number of proper nouns in a text"""
        if "nb_proper_nouns" in list(self.statistics.keys()):
            if self.statistics["nb_proper_nouns"] == None:
                self.statistics["nb_proper_nouns"] = self.readability_processor.count_proper_nouns(self.content,mode)
        else: 
            self.statistics["nb_proper_nouns"] = self.readability_processor.count_proper_nouns(self.content,mode)
        return self.statistics["nb_proper_nouns"]

    def lexical_cohesion_tfidf(self, mode="text", force=False):
        """
        Returns the average cosine similarity between adjacent sentences in a text after TFIDF representation.

        This can be done by representing the contents of each sentence in a term frequency-inverse document frequency matrix,
        and using that to calculate the cosine similarity between each represented sentence.

        By using the 'mode' parameter, can use inflected forms of tokens or their lemmas, possibly filtering the text beforehand
        in order to keep only nouns, proper names, and pronouns.
        Valid values for mode are : 'text', 'lemma', 'subgroup_text', 'subgroup_lemma'.

        :param str mode: Whether to filter the text, and whether to use raw texts or lemmas.
        :return: Average of cosine similarity between each adjacent sentence [i, i+1]
        :rtype: float
        """
        return self.call_score("cosine_similarity_tfidf",[mode],force)

    # NOTE: this seems to output the same values, whether we use text or lemmas, probably due to the type of model used.
    def lexical_cohesion_LDA(self ,mode="text", force=False):
        """
        Returns the average cosine similarity between adjacent sentences in a text by using a Latent Dirichlet allocation.

        This is a step further than the TFIDF method since this instead relates "topics" together instead of simply indicating
        whether two sentences share some exact words.
        This is done thanks to GenSim and Word2Vec : By first converting a text's sentences into BOW vectors,
        then by using the model to see if two adjacent sentences share the same topics.
        
        By using the 'mode' parameter, can use inflected forms of tokens or their lemmas.
        Valid values for mode are : 'text', 'lemma'.
        
        :param str mode: Whether to filter the text, and whether to use raw texts or lemmas.
        :return: Average of cosine similarity between each adjacent sentence [i, i+1]
        :rtype: float
        """
        return self.call_score("cosine_similarity_LDA",[mode],force)

        
