"""
The Readability module interacts with the library's submodules to provide a bunch of useful functions and scores for estimating the readability of a text.
By default, each score is available, but the user can choose to exclude some.

This module contains two types of components:
First, the low-level processor, which is referenced in the documentation or in the code as a 'readability processor'.
At start-up : It loads or imports external resources depending on which scores were kept.
Afterwards, this processor can use these resources in order to calculate the scores the user wants.

The second type are parsed texts, or collections of these texts, obtained by calling the parse or parseCollection functions from the readability processor.
On top of containing the text itself, these store the scores after calculations, and also store a bundle of informations can be used by multiple functions
in order to accelerate the process.
Furthermore, parsed collections can use additional functions that necessitate multiple texts, for instance : text classification can be attempted.
Please view the rest of the documentation for more details. 
"""

import spacy
from .utils import utils
from .stats import diversity, perplexity, common_scores, word_list_based, syntactic, discourse, rsrs
from .methods import methods
from .models import bert, fasttext, models
from .parsed_text import parsed_text
from .parsed_collection import parsed_collection

# Checklist :
#     Remake structure to help differenciate between functions : V Should be fine
#     Enable a way to "compile" in order to use underlying functions faster : ~ Done, need to modify underlying functions to take advantage of that when possible.
#     Make sure code works both for texts (strings, or ide a function that converts "anything" to a corpus structure) : V Done, with the convert... functions in .utils
#     Add the methods related to machine learning or deep learning) and corpus structure : ~ Detail documentation further
#     Add examples to the notebook to show how it can be used : ~ Done, need feedback now (and add more examples)
#     Add other measures that could be useful (Martinc | Crossley): ~ This is taking more time than expected since I'm also trying to understand what these do and why use them
#     Experiment further : X Unfortunately, I haven't given much thought into how estimating readability could be improved, or if our hypotheses are sound.

# For now :

#     Continue developping discourse/cohesion/coherence features. : X finish coherence already, and maybe a bit of syntactic.
#     Permettre de calculer scores|correlations en + automatiquement (Calculer scores de corr pour features cohesion (1er corpus minimum)) : Done.
#     Ajouter mesure de semi-partial correlation : X

# For today : tentative
# Actually finish the discourse module
# Fix the div 0 error in cosine similarities
# Expand readme and the notebook
# Work on synctactic features a bit?

# Extra (not urgent) :
#     Add more corpuses such as vikidia or wikimini : X (will probably start june 22 afternoon) :
#     I suppose I could crawl a bunch of articles on wikimini, and search for them on Wikipedia, hopefully getting a match.
#     ~300-500 articles should be enough.


# FIXME : several formulas are incorrect, as outlined in the submodule stats/common_scores.
# These being GFI, ARI due to wrong formulas, SMOG due to an error in calculating polysyllables, FRE due to a wrong variable assignation.
# For now, we kept these as is, in order to keep the paper's experiments reproducible

class Readability:
    """
    The Readability class provides a way to access the underlying library submodules in order to help estimate the complexity of any given text.
    At start-up : It loads or imports external resources depending on which scores were kept.
    Afterwards, this processor can use these resources in order to calculate the scores the user wants.

    - List of **attributes**::
        :param str lang: Placeholder : Language the text was written in, in order to adapt some scores.
        :param str nlp: Type of NLP processor to use, tentatively indicated with a "type_subtype" string.
        :param dict informations: Dictionary associating scores with the functions needed to calculate them, alongside the dependencies needed.
        :param dict excluded_informations: Same as above, but contains scores that have been excluded at start-up.
        :param dict dependencies: Dictionary associating dependency name with whatever is needed, usually a language model and its parameters.
    """
    def __init__(self, exclude = [""], lang = "fr", nlp = "spacy_sm"):
        """
        Constructor of the Readability class, won't return any value but creates the attributes :

        :param list(str) exclude: List of scores to exclude, in order to modify the `informations` and `dependencies` attributes.
        :param str lang: Placeholder : Language the text was written in, in order to adapt some scores.
        :param str nlp: Type of NLP processor to use, tentatively indicated with a "type_subtype" string.
        """
        self.lang = lang
        
        # Handle the NLP processor (mainly for tokenization in case we're given a text as a string)
        # FIXME : I tried adding the spacy model as a dependency in setup.cfg:
        # fr_core_news_sm@https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.3.0/fr_core_news_sm-3.3.0.tar.gz#egg=fr_core_news_sm
        # But I can't figure out how to use it, so this is a workaround.
        print("Acquiring Natural Language Processor...")
        if lang == "fr" and nlp == "spacy_sm":
            try:
                self.nlp = spacy.load('fr_core_news_sm')
                print("DEBUG: Spacy model location (already installed): ", self.nlp._path)
            except OSError:
                print('Downloading spacy language model \n(Should only happen once)')
                from spacy.cli import download
                download('fr_core_news_sm')
                self.nlp = spacy.load('fr_core_news_sm')
                print("DEBUG: Spacy model location: ", self.nlp._path)
        else:
            print("ERROR : Natural Language Processor not found for parameters : lang=",lang," nlp=",nlp,sep="")
            raise RuntimeError("ERROR : Natural Language Processor not found for parameters : lang=",lang," nlp=",nlp,sep="")
        
        # This dictionary associates values with the functions used to calculate them, alongside the dependencies needed.
        self.informations = dict(
            gfi=dict(function=self.gfi,dependencies=[],default_arguments=dict()),
            ari=dict(function=self.ari,dependencies=[],default_arguments=dict()),
            fre=dict(function=self.fre,dependencies=[],default_arguments=dict()),
            fkgl=dict(function=self.fkgl,dependencies=[],default_arguments=dict()),
            smog=dict(function=self.smog,dependencies=[],default_arguments=dict()),
            rel=dict(function=self.rel,dependencies=[],default_arguments=dict()),

            pppl=dict(function=self.perplexity,dependencies=["GPT2_LM"],default_arguments=dict()),

            ttr=dict(function=self.ttr,dependencies=[],default_arguments=dict(formula_type = "default")),
            ntr=dict(function=self.ntr,dependencies=[],default_arguments=dict(formula_type = "default")),

            dubois_buyse_ratio=dict(function=self.dubois_proportion,dependencies=["dubois_dataframe"],default_arguments=dict(filter_type="total",filter_value=None)),
            old20=dict(function=self.old20,dependencies=["lexique_dataframe"],default_arguments=dict()),
            pld20=dict(function=self.pld20,dependencies=["lexique_dataframe"],default_arguments=dict()),
            
            cosine_similarity_tfidf=dict(function=self.lexical_cohesion_tfidf,dependencies=[],default_arguments=dict(mode="text")),
            cosine_similarity_LDA=dict(function=self.lexical_cohesion_LDA,dependencies=["fauconnier_model"],default_arguments=dict(mode="text")),
            entity_density=dict(function=self.entity_density,dependencies=["coreferee"],default_arguments=dict(unique=False)),
            referring_entity_ratio=dict(function=self.proportion_referring_entity,dependencies=["coreferee"],default_arguments=dict()),
            average_entity_word_length=dict(function=self.average_word_length_per_entity,dependencies=["coreferee"],default_arguments=dict()),
            count_type_mention=dict(function=self.count_type_mention,dependencies=["coreferee"],default_arguments=dict(mention_type="proper_name")),
            count_type_opening=dict(function=self.count_type_opening,dependencies=["coreferee"],default_arguments=dict(mention_type="proper_name"))
            #following aren't implemented yet:
            #rsrs=dict(function=self.stub_rsrs,dependencies=["GPT2_LM"]),
            
            # TODO: These are meant to be used with a corpus only, so they should appear for a ParsedCollection instance, but not ParsedText.
            # However, I currently don't know how to implement that.
            #SVM_mean_accuracy=dict(function=self.classify_corpus_SVM,dependencies=[]),
            #MLP_mean_accuracy=dict(function=self.classify_corpus_MLP,dependencies=[]),
            #bert_metrics=dict(function=self.classify_corpus_BERT,dependencies=["BERT"]),
            #fasttext_metrics=dict(function=self.classify_corpus_fasttext,dependencies=["fastText"]),
            
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
            self.dependencies[dependency] = utils.load_dependency(dependency,self.nlp)


    # Utility functions : parse/load/checks
    def parse(self,text):
        """Returns a ParsedText instance, containing the text and a reference to the processor used, providing a way to store and output readability measures""" 
        return parsed_text.ParsedText(text,self)
    
    def parseCollection(self,collection):
        """
        Creates a ParsedCollection instance that relies on the ReadabilityProcessor in order to output various readability measures.

        Currently, three types of structures will be recognized as a collection of texts, albeit they'll be converted to the first format:
        A corpus-like dictionary that associates labels with texts. e.g : dict(class_1:{text1,text2},class_2:{text1,text2}).
        A list of lists of texts, given labels for compatibility with other functions.
        A singular list of texts, given a label for compatibility with other functions.
        """
        # Structure is dictionary, try to adapt the structure to be : dict(class_1:{text1,text2},class_2{text1,text2}..)
        if isinstance(collection,dict):
            copy_collection = dict()
            for label,text_list in collection.items():
                copy_list = []
                for text in text_list:
                    copy_list.append(self.parse(text))
                copy_collection[label] = copy_list
            return parsed_collection.ParsedCollection(copy_collection, self)
        elif isinstance(collection, list):
            try:
                # Check if collection contains a list of texts or a list of lists of texts
                # This raises an exception if not applied on a text, which means that we're currently handling a list containing lists of texts
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
                    self.dependencies[dependency] = utils.load_dependency(dependency,self.nlp)

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
        """
        Outputs pseudo-perplexity, which is derived from pseudo-log-likelihood scores.
        Please refer to this paper for more details : https://doi.org/10.18653%252Fv1%252F2020.acl-main.240

        :return: The pseudo-perplexity measure for a text, or for each text in a corpus.
        :rtype: float
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

        if not self.check_score_and_dependencies_available(name):
            raise RuntimeError("measure", name, "cannot be calculated.")
        if statistics is not None:
            return func(content, statistics)
        else:
            return func(content)
        
    def gfi(self, content, statistics = None):
        """
        Outputs the Gunning fog index, a 1952 readability test estimating the years of formal education needed to understand a text on the first reading.

        The scale goes from 6 to 18, starting at the sixth grade in the United States.
        The formula is : 0.4 * ( (words/sentences) + 100 * (complex words / words) )
        """
        return self.score("gfi", content, statistics)

    def ari(self, content, statistics = None):
        """
        Outputs the Automated readability index, a 1967 readability test estimating the US grade level needed to comprehend a text.

        The scale goes from 1 to 14, corresponding to age 5 to 18.
        The formula is 4.71 * (characters / words) + 0.5 (words / sentences) - 21.43
        """
        return self.score("ari", content, statistics)

    def fre(self, content, statistics = None):
        """
        Outputs the Flesch reading ease, a 1975 readability test estimating the US school level needed to comprehend a text.

        The scale goes from 100 to 0, corresponding to Grade 5 at score 100, up to post-college below score 30.
        The formula is 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
        """
        return self.score("fre", content, statistics)

    def fkgl(self, content, statistics = None):
        """
        Outputs the Flesch–Kincaid grade level, a 1975 readability test estimating the US grade level needed to comprehend a text.

        The scale is meant to be a one to one representation, a score of 5 means that the text should be appropriate for fifth graders.
        The formula is 0.39 * (total words / total sentences)+11.8*(total syllables / total words) - 15.59
        """
        return self.score("fkgl", content, statistics)

    def smog(self, content, statistics = None):
        """
        Outputs the Simple Measure of Gobbledygook, a 1969 readability test estimating the years of education needed to understand a text.

        The scale is meant to be a one to one representation, a score of 5 means that the text should be appropriate for fifth graders.
        The formula is 1.043 * Square root (Number of polysyllables * (30 / number of sentences)) + 3.1291
        """
        return self.score("smog", content, statistics)

    def rel(self, content, statistics = None):
        """
        Outputs the Reading Ease Level, an adaptation of Flesch's reading ease for the French language.

        The changes to the coefficients take into account the difference in length between French and English words.
        The formula is 207 - 1.015 * (Number of words / Number of sentences) - 73.6 * (Number of syllables / Number of words)
        """
        return self.score("rel", content, statistics)

    # Measures related to perplexity
    def perplexity(self,content):
        """
        Outputs pseudo-perplexity, which is derived from pseudo-log-likelihood scores.
        
        Please refer to this paper for more details : https://doi.org/10.48550/arXiv.1910.14659

        :return: The pseudo-perplexity measure for a text
        :rtype: float
        """
        if not self.check_score_and_dependencies_available("pppl"):
            raise RuntimeError("measure 'pppl' cannot be calculated, please try ReadabilityProcessor.load('pppl') and try again.")
        #print("Please be patient, pseudo-perplexity takes a lot of time to calculate.")
        return perplexity.PPPL_score(self.dependencies["GPT2_LM"],content)
    
    def stub_rsrs():
        #TODO : check submodule stats/rsrs for implementation details
        print("not implemented yet")
        return -1
    

    # Measures related to text diversity
    def diversity(self, content, ratio_type, formula_type=None):
        """
        Outputs a measure of text diversity based on which feature to use, and which version of the formula is used.

        Default formula is "nb lexical items / nb unique lexical items",
        'root' formula applies the square root to the denominator,
        'corrected' formula mutliplies the number of words by two before applying the square root to the denominator.

        :param str ratio_type: Which text diversity measure to use: "ttr" is text token ratio, "ntr" is noun token ratio
        :param str formula_type: What kind of formula version to use: "corrected", "root", and default standard are available for token ratios.
        :return: a measure of text diversity, or a dictionary of these measures
        :rtype: float
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

        They represent the mean Levenshtein distance between a word and its 20 closest neighbours:
        OLD20 is an alternative to the orthographical neighbourhood index that has been shown to correlate with text difficulty,
        due to being related to the perceptual ambiguity of word recognition when there exists close orthographic neighbours.
        This is the case for a low OLD20 value.
        
        Currently using the Lexique 3.0 database for French texts, version 3.83. More details here : http://www.lexique.org/

        :param str type: What kind of value to return, OLD20 or PLD20.
        :return: Average of OLD20 or PLD20 for each word in current text
        :rtype: float
        """
        func = word_list_based.average_levenshtein_distance
        if not self.check_score_and_dependencies_available(mode):
            raise RuntimeError("measure", mode, "cannot be calculated.")
        return func(self.dependencies["lexique_dataframe"]["dataframe"],content,self.nlp,mode)

    def old20(self, content):
        """Returns Orthographic Levenshtein distance for each word in a text."""
        return self.average_levenshtein_distance(content, "old20")

    def pld20(self, content):
        """Returns Phonemic Levenshtein distance for each word in a text."""
        return self.average_levenshtein_distance(content, "pld20")
        
    # Measures related to text cohesion :
    # NOTE : These 3 could be grouped together in the same function, and just set an argument type="X"
    def count_pronouns(self, content, mode="text"):
        """Returns number of pronouns in a text"""
        func = discourse.nb_pronouns
        return func(content,self.nlp,mode)
    
    def count_articles(self, content, mode="text"):
        """Returns number of articles in a text"""
        func = discourse.nb_articles
        return func(content,self.nlp,mode)
        
    def count_proper_nouns(self, content, mode="text"):
        """Returns number of proper nouns in a text"""
        func = discourse.nb_proper_nouns
        return func(content,self.nlp,mode)

    def lexical_cohesion_tfidf(self, content, mode="text"):
        """
        Returns the average cosine similarity between adjacent sentences in a text.

        By using the 'mode' parameter, can use inflected forms of tokens or the corresponding lemmas, possibly filtering the text beforehand
        in order to keep only nouns, proper names, and pronouns.
        Valid values for mode are : 'text', 'lemma', 'subgroup_text', 'subgroup_lemma'.

        :param str mode: Whether to filter the text, and whether to use raw texts or lemmas.
        :return: a ratio of words in the current text, that appear in the Dubois-Buyse word list.
        :rtype: float
        """
        if not self.check_score_and_dependencies_available("cosine_similarity_tfidf"):
            raise RuntimeError("measure 'cosine_similarity_tfidf' cannot be calculated.")
        func = discourse.average_cosine_similarity_tfidf
        return func(content,self.nlp,mode)

    # NOTE: this seems to output the same values, whether we use text or lemmas, probably due to the type of model used.
    def lexical_cohesion_LDA(self, content, mode="text"):
        """
        Returns the average cosine similarity between adjacent sentences in a text.
        
        By using the 'mode' parameter, can use inflected forms of words or their lemmas.
        Valid values for mode are : 'text', 'lemma'.
        """
        if not self.check_score_and_dependencies_available("cosine_similarity_LDA"):
            raise RuntimeError("measure 'cosine_similarity_LDA' cannot be calculated.")
        func = discourse.average_cosine_similarity_LDA
        return func(self.dependencies["fauconnier_model"],content,self.nlp,mode)

    def entity_density(self,content,unique=False):
        if not self.check_score_and_dependencies_available("entity_density"):
            raise RuntimeError("measure", "entity_density", "cannot be calculated.")
        func = discourse.entity_density
        return func(content,self.nlp,unique)

    def unique_entity_density(self,content):
        return self.entity_density(content=content,unique=True)

    def proportion_referring_entity(self,content):
        if not self.check_score_and_dependencies_available("referring_entity_ratio"):
            raise RuntimeError("measure", "referring_entity_ratio", "cannot be calculated.")
        func = discourse.proportion_referring_entity
        return func(content,self.nlp)

    def average_word_length_per_entity(self,content):
        if not self.check_score_and_dependencies_available("average_entity_word_length"):
            raise RuntimeError("measure", "average_entity_word_length", "cannot be calculated.")
        func = discourse.average_word_length_per_entity
        return func(content,self.nlp)

    def count_type_mention(self,content,mention_type="proper_name"):
        if not self.check_score_and_dependencies_available("count_type_mention"):
            raise RuntimeError("measure", "count_type_mention", "cannot be calculated.")
        func = discourse.count_type_mention
        return func(content,mention_type,self.nlp)
    
    #TODO : finish listing each possible variant one by one..
    def count_type_mention_proper_name(self,content):
        return self.count_type_mention(content,"proper_name")

    def count_type_opening(self,content,mention_type="proper_name"):
        if not self.check_score_and_dependencies_available("count_type_opening"):
            raise RuntimeError("measure", "count_type_opening", "cannot be calculated.")
        func = discourse.count_type_opening
        return func(content,mention_type,self.nlp)

    def count_type_opening_proper_name(self,content):
        return self.count_type_opening(content,"proper_name")


    # NOTE: the following methods are intended to be used with a corpus
    # Measures obtained from Machine Learning models :
    # TODO: allow user to optionally also use currently known features from other methods(common_scores, text diversity, etc..)
    def corpus_classify_ML(self,model_name,collection,plot=False):
        if model_name == "SVM":
            func = methods.classify_corpus_SVM
        elif model_name == "MLP":
            func = methods.classify_corpus_MLP
        elif model_name == "compare":
            func = methods.compare_models

        if isinstance(collection, parsed_collection.ParsedCollection):
            return func(collection, plot)
        else:
            if isinstance(collection,dict):
                return func(collection,plot)
            elif isinstance(collection, list):
                try:
                    # Check if collection contains a list of texts or a list of lists of texts
                    # This raises an exception if not applied on a text, which means that we're currently handling a list containing lists of texts
                    utils.convert_text_to_string(collection[0])
                except Exception:
                    # Case with multiple lists of texts:
                    counter = 0
                    copy_collection = dict()
                    for text_list in collection:
                        copy_list = []
                        for text in text_list:
                            copy_list.append(text)
                        copy_collection["label" + str(counter)] = copy_list
                        counter +=1
                    return func(copy_collection,plot)
                else:
                    raise TypeError("Cannot use a collection containing only one class for classification purposes, please try with something else.")
        return None

    # Machine Learning applications:
    def classify_corpus_SVM(self ,collection, plot=False):
        return self.corpus_classify_ML("SVM",collection,plot)

    def classify_corpus_MLP(self, collection, plot=False):
        return self.corpus_classify_ML("MLP",collection,plot)

    def compare_ML_models(self, collection, plot=True):
        return self.corpus_classify_ML("compare",collection,plot)

    # Deep Learning applications: 
    def classify_corpus_fasttext(self, collection, model_name = "fasttext"):
        func = fasttext.classify_corpus_fasttext
        return func(collection, model_name)
        
    def classify_corpus_BERT(self, collection, model_name = "camembert-base"):
        func = bert.classify_corpus_BERT
        return func(collection, model_name)
