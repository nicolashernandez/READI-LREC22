"""
The Readability module interacts with the library's submodules to provide a bunch of useful functions and scores for estimating the readability of a text
It will eventually support different languages.

This module provides the following services :
At start-up : it "converts" a text or corpus into a structure useful for the other functions, and can also calculate relevant statistics via the .compile function (number of words, sentences, syllables, etc..)
Calculating scores can be done without compiling beforehand, but is recommended in order to store information/scores each time a function is used, which can help speed up other functions.
This module also lazily loads external resources for more complicated things, like calculating perplexity or the use of Machine Learning / Deep Learning models
"""
import copy
import math

import pandas as pd
import spacy
from .utils import utils
from .stats import diversity, perplexity, common_scores, word_list_based, syntactic, discourse, rsrs
from .methods import methods
from .models import bert, fasttext, models
from .parsed_text import parsed_text
from .parsed_collection import parsed_collection

# Checklist :
#     Remake structure to help differenciate between functions : ~ I think it's okay but I need some feedback
#     Enable a way to "compile" in order to use underlying functions faster : V It's done, I should implement tests.
#     Make sure code works both for texts (strings, or pre-tokenized texts) and corpus structure : ~ I think it works now, need to provide a function that converts "anything" to a corpus structure
#     Add the methods related to machine learning or deep learning : ~ Need to make slightly different version for users (and document them)
#     Add examples to the notebook to show how it can be used : ~ Done, need feedback now
#     Add other measures that could be useful (Martinc | Crossley): ~ This is taking more time than expected since I'm also trying to understand what these do and why use them
#     Experiment further : X Unfortunately, I haven't given much thought into how estimating readability could be improved, or if our hypotheses are sound.

# For now :
#     Fix notebook.. again. I used things like os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../..', 'data'))
#     to reference the data folder that is a parent of the lib, but some user environments
#     refer to the lib via a '/usr/local/lib/python3.7/....' path (or something similar since it is installed..).
#     So it fails since the data folder isn't included in the library.
#     Either duplicate data inside the lib or apply a band-aid fix to the notebook (Not a good idea)

#     Continue developping discourse/cohesion/coherence features.
#     Ajouter une fonction pour générer tableau récap des features disponibles (ou celles déja calculées aprés .compile())
#     Permettre de calculer scores|correlations en + automatiquement (Calculer scores de corr pour features cohesion (1er corpus minimum))
#     Ajouter mesure de semi-partial correlation
#     Factoriser appels NLP/models pour être sûr qu'on ne duplique pas de ressources
#     Séparer initialisation object Readi | Parsing de texte
# parse lis un dict measures (ce dict measures contient ce qui est dispo selon l'init du readability processor, un dict avec 3 truc : 'possible', 'pas_possible')

# Extra (not urgent) :
#     Add more corpuses such as vikidia or wikimini : X (will probably start june 22 afternoon) :
#     I suppose I could crawl a bunch of articles on wikimini, and search for them on Wikipedia, hopefully getting a match.
#     ~300-500 articles should be enough.


# FIXME : several formulas are incorrect, as outlined in the submodule stats/common_scores.
# These being GFI, ARI due to wrong formulas, SMOG due to an error in calculating polysyllables, FRE due to a wrong variable assignation.
# For now, we kept these as is, in order to keep the paper's experiments reproducible

class Readability:
    """
    The Readability class provides a way to access the underlying library modules in order to help estimate the complexity of any given text
    List of methods : __init__, compile, corpus_info, stats, remove_outliers, score(score_name), scores, perplexity, do_function_with_default_arguments, get_corpus_scores, dubois_proportion, average_levenshtein_distance,
    List of attributes : content, content_type, lang, nlp, perplexity_processor, classes, statistics, corpus_statistics

    In its current state, scores are meant to be used with the French language, but this can change in the future.
    """
    def __init__(self, exclude = [""], content="dummy", lang = "fr", nlp = "spacy_sm"):
        """
        Constructor of the Readability class, won't return any value but creates the attributes :
        self.content, self.content_type, self.nlp, self.lang, and self.classes only if input is a corpus.

        :param content: Content of a text, or a corpus
        :type content: str, list(str), list(list(str)), converted into list(list(str)) or dict[class][text][sentence][token]
        :param list(str) exclude: List of type of scores to exclude, compared to the informations object to possibly remove unused dependencies
        :param str lang: Language the text was written in, in order to adapt some scores.
        :param str nlp: Type of NLP processor to use, indicated by a "type_subtype" string.
        :param str perplexity_processor: Type of processor to use for the calculation of pseudo-perplexity
        """
        self.lang = lang
        
        # Handle the NLP processor (mainly for tokenization in case we're given a text as a string)
        # Note : I tried adding the spacy model as a dependency in setup.cfg:
        # fr_core_news_sm@https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.3.0/fr_core_news_sm-3.3.0.tar.gz#egg=fr_core_news_sm
        # But I can't figure out how to use it, so this is a workaround.
        print("Acquiring Natural Language Processor...")
        if lang == "fr" and nlp == "spacy_sm":
            try:
                self.nlp = spacy.load('fr_core_news_sm')
                print("DEBUG: Spacy model location (already installed) : ", self.nlp._path)
            except OSError:
                print('Downloading spacy language model \n(Should only happen once)')
                from spacy.cli import download
                download('fr_core_news_sm')
                self.nlp = spacy.load('fr_core_news_sm')
                print("DEBUG: Spacy model location : ", self.nlp._path)
        else:
            print("ERROR : Natural Language Processor not found for parameters :  lang=",lang," nlp=",nlp,sep="")
            self.nlp = None
        
        # Handling text that needs to be converted into lists of tokens
        self.content = utils.convert_text_to_sentences(content,self.nlp)

        # Handling what is probably a corpus
        # Reminder, structure needed is : dict => list of texts => list of sentences => list of words
        # TODO : check this with a bunch of edge cases
        #if type(content) == dict:
        #    if isinstance(content[list(content.keys())[0]], list):
        #        if isinstance(content[list(content.keys())[0]][0], list):
        #            if isinstance(content[list(content.keys())[0]][0][0], list):
        #                self.content_type = "corpus"
        #                self.content = content
        #                self.classes = list(content.keys())
        #    #else, check if the structure is dict[class][text].. and tokenize everything (warn user it'll take some time)
        #    #and then use that as the new structure
        
        # This dictionary associates values with the functions used to calculate them, alongside the dependencies needed.
        self.informations = dict(
            gfi=dict(function=self.gfi,dependencies=[],default_arguments=dict()),
            ari=dict(function=self.ari,dependencies=[],default_arguments=dict()),
            fre=dict(function=self.fre,dependencies=[],default_arguments=dict()),
            fkgl=dict(function=self.fkgl,dependencies=[],default_arguments=dict()),
            smog=dict(function=self.smog,dependencies=[],default_arguments=dict()),
            rel=dict(function=self.rel,dependencies=[],default_arguments=dict()),
            pppl=dict(function=self.perplexity,dependencies=["GPT2_LM"],default_arguments=dict()),
            dubois_buyse_ratio=dict(function=self.dubois_proportion,dependencies=["dubois_dataframe"],default_arguments=dict(filter_type="total",filter_value=None)),
            ttr=dict(function=self.ttr,dependencies=[],default_arguments=dict(formula_type = "default")),
            ntr=dict(function=self.ntr,dependencies=[],default_arguments=dict(formula_type = "default")),
            old20=dict(function=self.old20,dependencies=["lexique_dataframe"],default_arguments=dict()),
            pld20=dict(function=self.pld20,dependencies=["lexique_dataframe"],default_arguments=dict()),
            cosine_similarity_tfidf=dict(function=self.lexical_cohesion_tfidf,dependencies=[],default_arguments=dict(mode="text")),
            cosine_similarity_LDA=dict(function=self.lexical_cohesion_LDA,dependencies=["fauconnier_model"],default_arguments=dict(mode="text"))
            #following aren't 100% implemented yet
            #bert_value=dict(function=self.stub_BERT,dependencies=["BERT"]),
            #fasttext_value=dict(function=self.stub_fastText,dependencies=["fastText"]),
            #rsrs=dict(function=self.stub_rsrs,dependencies=["GPT2_LM"]),
        )
        self.excluded_informations = dict()

        # Then remove things in self.informations based on what's in the exclude argument
        for value in list(self.informations.keys()):
            if value in exclude:
                self.excluded_informations[value] = self.informations[value]
                del self.informations[value]
        
        # Then iterate over what's remaining in self.informations to see what dependencies are needed:
        dependencies_to_add = set()
        for information in self.informations.values():
            for dependency in information["dependencies"]:
                dependencies_to_add.add(dependency)

        # Create a dependencies dictionary, and put what's needed in there after loading the external ressources
        self.dependencies = {}
        for dependency in dependencies_to_add:
            self.dependencies[dependency] = utils.load_dependency(dependency)


    # Utility functions : parse/load/checks
    def parse(self,text):
        """Creates a ParsedText instance that relies on the ReadabilityProcessor in order to output various readability measures."""
        return parsed_text.ParsedText(text,self)
    
    def parseCollection(self,collection):
        """
        Creates a ParsedCollection instance that relies on the ReadabilityProcessor in order to output various readability measures.
        Currently, only three types of structures are accepted:
        A corpus-like dictionary that associates labels with texts. e.g : dict(class_1:{text1,text2},class_2:{text1,text2}).
        A list of lists of texts. Will be given labels for compatibility with other functions.
        A singular list of texts. Will be given a label for compatibility with other functions.
        """
        # Structure is dictionary, try to adapt the structure to be : dict(class_1:{text1,text2},class_2{text1,text2}..)
        if isinstance(collection,dict):
            return parsed_collection.ParsedCollection(collection, self)
        elif isinstance(collection, list):
            try:
                # Check if collection contains a list of texts or a list of lists of texts
                # This raises an exception if not applied on a text.
                utils.convert_text_to_string(collection[0])
            except Exception:
                # Case with multiple lists of texts:
                counter = 0
                copy_collection = dict()
                for text_list in collection:
                    copy_list = []
                    for text in text_list:
                        copy_list.append(self.parse(text))
                    copy_collection["label" + str(counter)] = copy_list
                    counter +=1
                return parsed_collection.ParsedCollection(copy_collection,self)
            else:
                # Case with one list of texts:
                copy_collection = []
                for text in collection:
                    copy_collection.append(self.parse(text))
                copy_collection = dict(label0 = copy_collection)
                return parsed_collection.ParsedCollection(copy_collection, self)

        else:
            raise TypeError("Format of received collection not recognized, please give dict(class_name:{list(text)}) or list(list(text))")

    #NOTE : maybe also provide load_dependency(self,value)
    def load(self,value):
        """Checks if a measure or value has been excluded, enables it and loads its dependencies if needed."""
        # Based on the value's name, check if exists in self.excluded_informations
        if value in list(self.excluded_informations.keys()):
            # Transpose back to self.informations
            self.informations[value] = self.excluded_informations[value]
            del self.excluded_informations[value]
            print("Value '",value,"' can now be calculated",sep="")
            # Check if there's a dependency, and handle it if wasn't imported already
            for dependency in self.informations[value]["dependencies"]:
                if dependency not in list(self.dependencies.keys()):
                    self.dependencies[dependency] = utils.load_dependency(dependency)

        elif value in list(self.informations.keys()):
            # Check if it's in self.informations to warn user it's already loaded
            print("No need to call .load(",value,"), value already exists in instance.informations[",value,"]",sep="")
            print(self.informations[value])
        else:
            # Raise error to tell user this measure isn't recognized
            raise ValueError("Value",value,"was not recognized as par of instance.informations or instance.excluded_informations, Please check if you've done a typo.")
    
    def check_score_and_dependencies_available(self,score_name):
        """Indicates whether a measure or value has been excluded, and if its dependencies are available."""
        if score_name not in list(self.informations.keys()):
            print("Value", score_name, "was not found in instance.informations. Please check if you excluded it when initializing the ReadabilityProcessor.")            
            return False
        else:
            if score_name in list(self.informations.keys()):
                dependencies = self.informations[score_name]["dependencies"]
            else:
                dependencies = self.excluded_informations[score_name]["dependencies"]
            for dependency_name in dependencies:
                if dependency_name not in list(self.dependencies.keys()):
                    print("Dependency", dependency_name, "was not found in instance.dependencies. Something's gone wrong")
                    return False
        return True


    # Traditional measures: 
    def score(self, name, content, statistics = None):
        """This function calls a score's calculation from the submodule common_scores"""
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

        if not self.check_score_and_dependencies_available(name):
            raise RuntimeError("measure", name, "cannot be calculated.")
        if statistics is not None:
            return func(content, statistics)
        else:
            return func(content)
        
    def gfi(self, content, statistics = None):
        """Returns Gunning Fog Index"""
        return self.score("gfi", content, statistics)

    def ari(self, content, statistics = None):
        """Returns Automated Readability Index"""
        return self.score("ari", content, statistics)

    def fre(self, content, statistics = None):
        """Returns Flesch Reading Ease"""
        return self.score("fre", content, statistics)

    def fkgl(self, content, statistics = None):
        """Returns Flesch–Kincaid Grade Level"""
        return self.score("fkgl", content, statistics)

    def smog(self, content, statistics = None):
        """Returns Simple Measure of Gobbledygook"""
        return self.score("smog", content, statistics)

    def rel(self, content, statistics = None):
        """Returns Reading Ease Level (Adaptation of FRE for french)"""
        return self.score("rel", content, statistics)

    #TODO : repurpose this inside ParsedText as a traditional-only version (for reproducing the paper's contents)
    # and a version with every available score
    def scores(self):
        """
        Depending on type of content provided, returns a list of common readability scores (type=text),
        or returns a matrix containing the mean values for these scores, depending on the classes of the corpus (type=corpus) 

        :return: a pandas dataframe (or a list of scores)
        :rtype: Union[pandas.core.frame.DataFrame,list] 
        """
        # NOTE : Need to rename this to something clearer, since we now have a method called "getScores"
        # NOTE : Would be better to have this point to a scores_text() and scores_corpus(), which returns only one type.
        # TODO : re-do this : show every available score if .compile() was done (and their pearson correlation)
        # Semi-partial correlation should also be available, but ask user beforehand since might need time to recalculate.
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


    # Measures related to perplexity
    def perplexity(self,content):
        """
        Outputs pseudo-perplexity, which is derived from pseudo-log-likelihood scores.
        Please refer to this paper for more details : https://doi.org/10.18653%252Fv1%252F2020.acl-main.240

        :return: The pseudo-perplexity measure for a text, or for each text in a corpus.
        :rtype: :rtype: Union[float,dict[str][list(float)]] 
        """
        if not self.check_score_and_dependencies_available("pppl"):
            raise RuntimeError("measure 'pppl' cannot be calculated, please try ReadabilityProcessor.load('pppl') and try again.")
        #print("Please be patient, pseudo-perplexity takes a lot of time to calculate.")
        return perplexity.PPPL_score(self.dependencies["GPT2_LM"],content)
    
    def stub_rsrs():
        #TODO : check submodule for implementation details
        print("not implemented yet")
        return -1
    

    # Measures related to text diversity
    def diversity(self, content, ratio_type, formula_type=None):
        """
        Outputs a measure of text diversity based on which feature to use, and which version of the formula is used.
        Default formula is "nb lexical items / nb unique lexical items",
        'root' formula uses the square root for the denominator,
        'corrected' formula mutliplies the number of words by two before taking the square root for the denominator.

        :param str ratio_type: Which text diversity measure to use: "ttr" is text token ratio, "ntr" is noun token ratio
        :param str formula_type: What kind of formula version to use: "corrected", "root", and default standard are available for token ratios.
        :return: a measure of text diversity, or a dictionary of these measures
        :rtype: :rtype: Union[float,dict[str][list(float)]]
        """
        if ratio_type == "ttr":
            func = diversity.type_token_ratio
        elif ratio_type == "ntr":
            func = diversity.noun_token_ratio

        if not self.check_score_and_dependencies_available(ratio_type):
            raise RuntimeError("measure", formula_type, "cannot be calculated.")
        return func(content, self.nlp, formula_type)

    def ttr(self, content, formula_type=None):
        """Returns Text Token Ratio: number of unique words / number of words"""
        return self.diversity(content, "ttr",formula_type)

    def ntr(self, content, formula_type=None):
        """Returns Noun Token Ratio: number of unique nouns / number of nouns"""
        return self.diversity(content, "ntr",formula_type)

    # Measures based on pre-existing word lists
    def dubois_proportion(self, content, filter_type = "total", filter_value = None):
        """
        Outputs the proportion of words included in the Dubois-Buyse word list.
        Can specify the ratio for words appearing in specific echelons, ages, or three-year cycles.

        :param str filter_type: Which variable to use to filter the word list : 'echelon', 'age', or 'cycle'
        :param str filter_value: Value (or iterable containing two values) for subsetting the word list.
        :type filter_value: Union[int, tuple, list]
        :return: a ratio of words in the current text, that appear in the Dubois-Buyse word list.
        :rtype: float
        """
        func = word_list_based.dubois_proportion
        if not self.check_score_and_dependencies_available("dubois_buyse_ratio"):
            raise RuntimeError("measure 'dubois_buyse_ratio' cannot be calculated.")
        return func(self.dependencies["dubois_dataframe"]["dataframe"], content, self.nlp, filter_type, filter_value)

    def average_levenshtein_distance(self, content, mode = "old20"):
        """
        Returns the average Orthographic Levenshtein Distance 20 (OLD20), or its phonemic equivalent (PLD20).
        Currently using the Lexique 3.0 database for French texts, version 3.83. More details here : http://www.lexique.org/
        OLD20 is an alternative to the orthographical neighbourhood index that has been shown to correlate with text difficulty.

        :param str type: What kind of value to return, OLD20 or PLD20.
        :return: Average of OLD20 or PLD20 for each word in current text
        :rtype: Union[float,dict[str][list(float)]]
        """
        func = word_list_based.average_levenshtein_distance
        if not self.check_score_and_dependencies_available(mode):
            raise RuntimeError("measure", mode, "cannot be calculated.")
        return func(self.dependencies["lexique_dataframe"]["dataframe"],content,self.nlp,mode)

    def old20(self, content):
        return self.average_levenshtein_distance(content, "old20")

    def pld20(self, content):
        return self.average_levenshtein_distance(content, "pld20")

    # NOTE : These 3 could be grouped together in the same function, and just set an argument type="X"
    def count_pronouns(self, content, mode="text"):
        func = discourse.nb_pronouns
        return func(content,self.nlp,mode)
    
    def count_articles(self, content, mode="text"):
        func = discourse.nb_articles
        return func(content,self.nlp,mode)
        
    def count_proper_nouns(self, content, mode="text"):
        func = discourse.nb_proper_nouns
        return func(content,self.nlp,mode)

    def lexical_cohesion_tfidf(self, content, mode="text"):
        if not self.check_score_and_dependencies_available("cosine_similarity_tfidf"):
            raise RuntimeError("measure 'cosine_similarity_tfidf' cannot be calculated.")
        func = discourse.average_cosine_similarity_tfidf
        return func(content,self.nlp,mode)

    # NOTE: this seems to output the same values, whether we use text or lemmas, probably due to the type of model used.
    def lexical_cohesion_LDA(self, content, mode="text"):
        if not self.check_score_and_dependencies_available("cosine_similarity_LDA"):
            raise RuntimeError("measure 'cosine_similarity_LDA' cannot be calculated.")
        func = discourse.average_cosine_similarity_LDA
        return func(self.dependencies["fauconnier_model"],content,self.nlp,mode)


    # Measures obtained from Machine Learning models :
    def stub_SVM():
        #TODO: allow user to use default tf-idf matrix thing or with currently known features from other methods(common_scores text diversity, etc..)
        return -1

    def stub_MLP():
        #TODO: allow user to use default tf-idf matrix thing or with currently known features from other methods(common_scores text diversity, etc..)
        return -1

    def stub_compareModels():
        #NOTE: this is probably not too high on the priority list of things to do.
        return -1


    # Measures obtained from Deep Learning models
    def stub_fastText():
        return -1
        
    def stub_BERT():
        return -1
