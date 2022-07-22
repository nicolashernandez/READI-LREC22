"""
The ParsedCollection module is a convenient way to group ParsedTexts together and enable additional functions that require group(s) of text by the processor.
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

    It is meant to be created as a result of ReadabilityProcessor.parseCollection() since it depends on a ReadabilityProcessor in order to access functions,
    and also supposes that each text stored is actually an instance of ParsedText.

    List of methods : __init__, call_score(), show_available_scores(), show_scores(), show_statistics(), remove_outliers()
    It also contains accessor functions based on ReadabilityProcessor methods, sharing the same name, these use the helper function call_score() in order to work.
    List of attributes : content, readability_processor, statistics, scores
    """
    def __init__(self, text_collection, readability_processor):
        """
        Constructor of the ParsedCollection class, creates the 'content', 'scores', 'statistics', and 'readability_processor' attributes.

        Keep in mind that the scores default to None since they haven't been calculated yet.

        :param dict(ParsedText) content: Structure associating labels with their associated texts. Defaults to a single dummy label called 'label0' if none are provided.
        :param dict scores: Language the text was written in, in order to adapt some scores.
        :param dict statistics: Common values used by various measures (Such as number of words, number of sentences, etc)
        :param str nlp: Type of NLP processor to use, indicated by a "type_subtype" string.
        :param ReadabilityProcessor readability_processor: Type of processor to use for the calculation of pseudo-perplexity
        """
        self.readability_processor = readability_processor
        self.content = text_collection

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
        Prints to the console the contents of the statistics for each class of text.
        """
        for label in list(self.content.keys()):
            print(label + "------------------")
            for stat in list(self.statistics[label].keys()):
                if stat == "vocabulary":
                    print(stat, "=", len(self.statistics[label][stat]), "words")
                else:
                    print(stat, "=", self.statistics[label][stat])

    def remove_outliers(self, score_type = None, stddevratio=1):
        """
        Outputs a corpus, after removing texts which are considered to be "outliers", based on a standard deviation ratio.
        A text is an outlier if its value is lower or higher than this : mean +- standard_deviation * ratio
        In order to exploit this new corpus, you'll need to make to parse the collection again.
        For instance : collection_without_GFI_outliers = ReadabilityProcessor.parseCollection(original_collection).remove_outliers("gfi",1)

        :param str score_type: A score type that can be recognized by the helpre function call_score (which relies on a ReadabilityProcessor's 'informations' attribute)
        :param float stddevratio: A number representing to which extent can be a value be tolerated before it is detected as an outlier: Less than mean - ratio, or more than mean + ratio.

        :return: a corpus, in a specific format where texts are represented as lists of sentences, which are lists of words.
        :rtype: dict[str][str][str][str]
        """
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
        return ParsedCollection(corpus_no_outliers,self.readability_processor)

    def call_score(self,score_name, arguments=None, force=False, iterable_arguments=None):
        """
        Helper function that gets a score if it already exists, otherwise checks if it's available, if so call the relevant function from the ReadabilityProcessor
        
        Use of function is : instance.call_score("score_name", arguments:[arg1,arg2,argi..], force:bool)
        If the underlying function needs no additional arguments, just pass en empty list, e.g : instance.call_score("pppl",[],True)
        Additionally, the function to be called can possibly use something that changes per class, or text in order to speed up the process.
        In this case, when using call_score, please pass these values in the argument 'iterable_arguments'
        as a list of lists, each list containing the values of the ith iterable argument for all texts.

        :param str score_name: Name of a score recognized by ReadabilityProcessor.informations.
        :param list(any) arguments: Values used to change behavior of underlying functions.
        :param bool force: Indicates whether to force the calculation of a score or not.
        :param list(list(any)) iterable_arguments: Additional values used to change behavior of underlying functions, on a text-by-text basis.
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
                if iterable_arguments is None:
                    for text in self.content[label]:
                        moy += text.call_score(score_name,arguments,force)
                else:
                    for index,text in enumerate(self.content[label]):
                        # Append every ith iterable argument to the permanent arguments (even if none are supplied)
                        # NOTE: this breaks if arguments is None, remember to make sure arguments is an empty list instead.
                        moy += text.call_score(score_name,arguments + list(list(zip(*iterable_arguments))[index]),force)
                self.scores[score_name][label] = moy / len(self.content[label])
                moy_score[label] = self.scores[score_name][label]
            else:
                moy_score[label] = None
        return moy_score

    def show_scores(self,force=False,correlation=None):
        """
        Returns a dataframe containing each calculated score, can force calculation with default values, and add a correlation coefficient.
        
        :param bool force: Indicates whether to force the calculation of each score
        :param str correlation: What kind of correlation coefficient to add to the dataframe.
        """
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
    
    def show_available_scores(self):
        """Prints currently 'available' scores' names in a list"""
        return list(self.scores.keys())
    
    # Traditional measures :
    def traditional_score(self,score_name, force=False):
        """
        Called by methods : gfi | ari | fre | fkgl | smog | rel. Serves as a entry-point to function "traditional_score" of ReadabilityProcessor.
        
        :param str score_name: Name of a score recognized by ReadabilityProcessor.informations.
        :param bool force: Indicates whether to force the calculation of a score or not.
        """
        text_statistics = [[]]
        for label in list(self.content.keys()):
            for parsed_text in self.content[label]:
                text_statistics[0].append(parsed_text.statistics)

        return self.call_score(score_name,arguments=[],force=force,iterable_arguments=text_statistics)
    
    def gfi(self, force=False):
        """
        Outputs the Gunning fog index, a 1952 readability test estimating the years of formal education needed to understand a text on the first reading.
        The scale goes from 6 to 18, starting at the sixth grade in the United States.
        The formula is : 0.4 * ( (words/sentences) + 100 * (complex words / words) )
        """
        return self.traditional_score("gfi")

    def ari(self, force=False):
        """
        Outputs the Automated readability index, a 1967 readability test estimating the US grade level needed to comprehend a text
        The scale goes from 1 to 14, corresponding to age 5 to 18.
        The formula is 4.71 * (characters / words) + 0.5 (words / sentences) - 21.43
        """
        return self.traditional_score("ari")

    def fre(self, force=False):
        """
        Outputs the Flesch reading ease, a 1975 readability test estimating the US school level needed to comprehend a text
        The scale goes from 100 to 0, corresponding to Grade 5 at score 100, up to post-college below score 30.
        The formula is 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
        """
        return self.traditional_score("fre")

    def fkgl(self, force=False):
        """
        Outputs the Flesch–Kincaid grade level, a 1975 readability test estimating the US grade level needed to comprehend a text
        The scale is meant to be a one to one representation, a score of 5 means that the text should be appropriate for fifth graders.
        The formula is 0.39 * (total words / total sentences)+11.8*(total syllables / total words) - 15.59
        """
        return self.traditional_score("fkgl")

    def smog(self, force=False):
        """
        Outputs the Simple Measure of Gobbledygook, a 1969 readability test estimating the years of education needed to understand a text
        The scale is meant to be a one to one representation, a score of 5 means that the text should be appropriate for fifth graders.
        The formula is 1.043 * Square root (Number of polysyllables * (30 / number of sentences)) + 3.1291
        """
        return self.traditional_score("smog")

    def rel(self, force=False):
        """
        Outputs the Reading Ease Level, an adaptation of Flesch's reading ease for the French language,
        with changes to the coefficients taking into account the difference in length between French and English words.
        The formula is 207 - 1.015 * (Number of words / Number of sentences) - 73.6 * (Number of syllables / Number of words)
        """
        return self.traditional_score("rel")


    # Measures related to perplexity
    def perplexity(self,force=False):
        """
        Outputs pseudo-perplexity, which is derived from pseudo-log-likelihood scores.
        Please refer to this paper for more details : https://doi.org/10.18653%252Fv1%252F2020.acl-main.240

        :param bool force: Indicates whether to force the calculation of a score or not.
        :return: The pseudo-perplexity measure for a text, or for each text in a corpus.
        :rtype: float
        """
        return self.call_score("pppl",[],force)
    
    def stub_rsrs(self,force=False):
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
    def dubois_buyse_ratio(self,filter_type="total",filter_value=None, force=False):
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
        return self.call_score("dubois_buyse_ratio",[filter_type, filter_value],force)

    def average_levenshtein_distance(self,mode="old20", force=False):
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
    def lexical_cohesion_LDA(self, mode="text", force=False):
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

    # Following functions can only be used with a collection of text.
    def classify_corpus_SVM(self, plot=False):
        """Uses a SVM (Support Vector Machine) model to classify the given collection of texts."""
        return self.readability_processor.classify_corpus_SVM(self,plot)
    def classify_corpus_MLP(self, plot=False):
        """Uses a MLP (Multilayer perceptron) model to classify the given collection of texts."""
        return self.readability_processor.classify_corpus_MLP(self,plot)
    def compare_ML_models(self, plot=True):
        """
        Uses several popular Machine Learning models to classify the given collection of texts, to show which ones currently performs the best.

        Uses a Random Forest Classifier, a SVM (Support Vector Machine) model, a Multinomial Naive Bayes model, Logistic Regression, and a Multilayer
        perceptron to see which ones currently performs the text classifiction task the best.
        """
        return self.readability_processor.compare_ML_models(self,plot)

    def classify_corpus_fasttext(self, model_name="fasttext"):
        """
        Imports, configures, and trains a fastText model.

        :param corpus: Data input, preferably as a dict(class_label:list(text))
        :param str model_name: Choice of language model to use : fasttext, bigru, nbsvm
        :return: Classification task metrics, as detailed in .models.compute_evaluation_metrics()
        """
        return self.readability_processor.classify_corpus_fasttext(self, model_name)
    def classify_corpus_BERT(self, model_name="camembert-base"):
        """
        Imports, configures, and trains a BERT model.
        
        :param corpus: Data input, preferably as a dict(class_label:list(text))
        :param str model_name: Choice of language model to use : bert-base-multilingual-cased, camembert-base, flaubert/flaubert_base_cased
        :return: Classification task metrics, as detailed in .models.compute_evaluation_metrics()
        """
        return self.readability_processor.classify_corpus_BERT(self, model_name)

