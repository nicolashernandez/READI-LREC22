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
#     refer to the lib via a '/usr/local/lib/python3.7/....' path (or something similar since it is instlled..).
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


# 29/06/2022 : Main Task
#v0) 
# r = readability.Readability(text1) # gives a self.content for handling the text
#       pppl = r.perplexity()
#v1)
# r = readability.Readability()
#       # r = readability.Readability(text1) # gives a self.content for handling the text
#       r.parse(text1) # => in charge of all common processes shared by various measures
#       # => Returns object parsed_text_1
#       #Warning : traditional features and Perplexity and BERT-like classification do not share common processes.
#       pppl = r.perplexity()
#       # nv text :
#       r.parse(text2)
#       pppl_2 = r.perplexity()
#       # naive :
#       r.parse(text1)
#       gfi = r.gfi()
# 
#v2)
# readability_processor = readability.Readability(exclude = [“pppl”])
# pt1 = readability_processor.parse(text1) # ParsedText .r = R
# pppl = pt1.perplexity() # Not Available value
#     r.load(“pppl”) 
#       pt1.set(r) # ? besoin de cette méthode set ?
#     pppl = pt1.perplexity() # Available value
#       pt2 = r.parse(text2)
#       pppl_2 = pt2.perplexity()

#What to do when user calls same method multiple times? if it's just a getter then no problem..
#pt1.perplexity() 
#pt1.perplexity() 
#usage : pour 1 texte donné, appliquer toutes les mesures. Raisonnable.


# modify r.compile() to become r.parse()
# Usage of API should be : 
# 1) init (load necessary models, prep processor)
# 2) parse text 
# 3) use any measure (keep external ressources loaded in r if needed)
#
# 
#29/01
#la méthode nlp doit être appelée qu’une seule fois quelque soit le nombre de features demandées
#dupliquer Readability implique forcément dupliquer des ressources/traitement e.g. chargement des modèles 
# -> distinguer la création de Readability (qui va charger le nécessaire commun au traitement de 1 ou plusieurs textes) du traitement des textes (parse)
#usage : 
#pour un texte, calculer plusieurs mesures -> factoriser ce qui est commun à plusieurs mesures ~ parse 
#pour des textes différents, utiliser une même mesure -> éviter de répéter des traitements en double si indépendant 
#par défaut la création de l’objet Readability charge tous les modèles mais on peut en exclure au chargement ou bien en charger ultérieurement,
# s’inspirer de la méthode de chargement de spacy https://spacy.io/usage/processing-pipelines#disabling
#méthode parse/process/measure : applique toutes les mesures non exclues ; retourne un objet parsed text avec des accesseurs 
#permettant où toutes les mesures ont été appelés
#actuellement gère notion de corpus découpé en classe et calcul moyenne des scores pour une classe donnée -> éventuellement une méthode intermédiaire 
#non adhérente au format corpus (dict…) qui traite une mesure pour une liste de texte (sans notion de classe)  


# Fonctionnement de la librarie détaillée :

# D'abord d'initaliser un readability_processor en excluant certains modèles et en précisant la langue (bon seulement le français fonctionne pour l'instant).
# => readability_processor = readability.Readability(lang = "fr", exclude = ["pppl","other_score","abc"]):
# On suppose que l'on cherche ensuite chaque score indiquée dans un objet methods, qui associe chaque score à une fonction de calcul, et possiblement un modèle.
# Si detection, on transpose ces associations dans un objet excluded_methods.
# On regarde ensuite tout les modèles conservés dans l'objet methodes : d'aprés ce qui reste, le processeur charge les modèles et prépare le nécessaire.

# L'utilisateur décide de traiter un texte :
# => pt1 = readability_processor.parse(text1) #renvoie un objet parsedText, dont un attribut est une reference à readability_processor
# Je suppose que deux objets peuvent partager la même instance mais on verra lors du développement.
# Lors de l'execution de .parse(), pt1 génére une liste de mesures utiles pour différentes méthodes : reprise du code actuel r.compile()
# L'utilisateur peut faire pt1.show_values() et ça renvoie un tableau contenant tout les scores calculables, ou déja calculés :
# On regarde d'abord chaque score calculé dans une liste pt1.scores["score"]
# Et on regarde ce qui est disponible dans methods dans pt1.r, on décide de calculer ou non et d'ajouter à la liste NA ou NaN si cela n'est pas le cas.
# Pour le reste inclus dans excluded_methods, on ajoute à la liste en indiquant NA ou NaN.
# L'utilisateur peut décider de recupérer un score en particulier en appelant la fonction :
# => pppl = pt1.perplexity()
# Dans ce cas, on fait une vérification (if pt1.score['pppl'] == None : do the function otherwise return the score).
# De plus, on peut décider de faire r.load("pppl") pour rendre disponible la fonction (en chargeant éventuellement les ressources nécessaires)




# NOTE: There probably exists a better way to create an Statistics object as an attribute of Readability.
class Statistics:
    pass

class Readability:
    """
    The Readability class provides a way to access the underlying library modules in order to help estimate the complexity of any given text
    List of methods : __init__, compile, corpus_info, stats, remove_outliers, score(score_name), scores, perplexity, do_function_with_default_arguments, get_corpus_scores, dubois_proportion, average_levenshtein_distance,
    List of attributes : content, content_type, lang, nlp, perplexity_processor, classes, statistics, corpus_statistics

    In its current state, scores are meant to be used with the French language, but this can change in the future.
    """
    def __init__(self, content, lang = "fr", nlp = "spacy_sm", perplexity_processor = "gpt2"):
        """
        Constructor of the Readability class, won't return any value but creates the attributes :
        self.content, self.content_type, self.nlp, self.lang, and self.classes only if input is a corpus.

        :param content: Content of a text, or a corpus
        :type content: str, list(str), list(list(str)), converted into list(list(str)) or dict[class][text][sentence][token]
        :param str lang: Language the text was written in, in order to adapt some scores.
        :param str nlp: Type of NLP processor to use, indicated by a "type_subtype" string.
        :param str perplexity_processor: Type of processor to use for the calculation of pseudo-perplexity
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
        self.content_type = "text"
        self.content = utils.convert_text_to_sentences(content,self.nlp)

        # Handling corpus that probably is a corpus
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
        
        if self.content_type == "text":
            nb_words = 0
            for sentence in self.content:
                nb_words += len(sentence)
            if nb_words < 101:
                print("WARNING: Text length is less than 100 words, some scores will be inaccurate.")
        

        # This is a dictionary that maps values that can be obtained from the readability class with the functions used to calculate them.
        # This is used for generalizable methods such as remove_outliers can be used no matter what kind of score.
        # It differenciates between scores that are obtained directly from a function, or functions that need a parameter to indicate the type of score to return.
        self.score_types = dict(
            no_argument_needed = dict(
                gfi=self.gfi,
                ari=self.ari,
                fre=self.fre,
                fkgl=self.fkgl,
                smog=self.smog,
                rel=self.rel,
                perplexity=self.perplexity,
                dubois_buyse_ratio=self.dubois_proportion),
            argument_needed = dict(
                ttr=self.diversity,
                ntr=self.diversity,
                old20=self.average_levenshtein_distance,
                pld20=self.average_levenshtein_distance)
            )

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
            for level in self.classes:
                temp_score = 0
                for index,score in enumerate(scores[level]):
                    temp_score += score
                    if hasattr(self, "corpus_statistics"):
                        setattr(self.corpus_statistics[level][index],name,score)
                temp_score = temp_score / len(scores[level])
                print("class", level, "mean score :" ,temp_score)
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


    def perplexity(self):
        """
        Outputs pseudo-perplexity, which is derived from pseudo-log-likelihood scores.
        Please refer to this paper for more details : https://doi.org/10.18653%252Fv1%252F2020.acl-main.240

        :return: The pseudo-perplexity measure for a text, or for each text in a corpus.
        :rtype: :rtype: Union[float,dict[str][list(float)]] 
        """
        if not hasattr(self.perplexity_calculator, "model_loaded"):
            print("Loading model for pseudo-perplexity..")
            self.perplexity_calculator.load_model()
            print("Model is now loaded")
        print("Please be patient, pseudo-perplexity takes a lot of time to calculate.")
        if self.content_type == "text":
            return self.perplexity_calculator.PPPL_score_text(self.content)
        elif self.content_type == "corpus":
            scores = self.perplexity_calculator.PPPL_score(self.content)
            if hasattr(self, "corpus_statistics"):
                for level in self.classes:
                    for index,score in enumerate(scores[level]):
                        setattr(self.corpus_statistics[level][index],"perplexity",score)
            return scores
        return -1


    def diversity(self, type, mode=None):
        """
        Outputs a measure of text diversity based on which feature to use, and which version of the formula is used.
        If using this for a corpus instead of a text, returns a dictionary containing the relevant scores.
        
        :param str type: Which text diversity measure to use : "ttr" is text token ratio, "ntr" is noun token ratio
        :param str mode: What kind of formula version to use, if applicable : "corrected", "root", and default standard are available for token ratios.
        :return: a measure of text diversity, or a dictionary of these measures
        :rtype: :rtype: Union[float,dict[str][list(float)]]
        """
        if type == "ttr":
            func = diversity.type_token_ratio
        elif type == "ntr":
            func = diversity.noun_token_ratio

        if self.content_type == "text":
            return func(self.content, self.nlp, mode)
        elif self.content_type == "corpus":
            scores = {}
            for level in self.classes:
                temp_score = 0
                scores[level] = []
                for index,text in enumerate(self.content[level]):
                    scores[level].append(func(text, self.nlp, mode))
                    temp_score += scores[level][index]
                    if hasattr(self, "corpus_statistics"):
                        setattr(self.corpus_statistics[level][index],locals()['type'],scores[level][index])
                temp_score = temp_score / len(scores[level])
                print("class", level, "mean score :" ,temp_score)
            return scores


    def dubois_proportion(self, filter_type = "total", filter_value = None):
        """
        Outputs the proportion of words included in the Dubois-Buyse word list.
        Can specify the ratio for words appearing in specific echelons, ages|school grades, or three-year cycles.

        :param str filter_type: Which variable to use to filter the word list : 'echelon', 'age', or 'cycle'
        :param str filter_value: Value (or iterable containing two values) for subsetting the word list.
        :type filter_value: Union[int, tuple, list]
        :return: a ratio of words in the current text, that appear in the Dubois-Buyse word list.
        :rtype: Union[float,dict[str][list(float)]]
        """
        func = word_list_based.dubois_proportion
        if self.content_type == "text":
            return func(self.content,self.nlp,filter_type,filter_value)
        elif self.content_type == "corpus":
            scores = {}
            for level in self.classes:
                temp_score = 0
                scores[level] = []
                for index,text in enumerate(self.content[level]):
                    scores[level].append(func(text,self.nlp,filter_type,filter_value))
                    temp_score += scores[level][index]
                    if hasattr(self, "corpus_statistics"):
                        self.corpus_statistics[level][index].dubois_buyse_ratio = scores[level][index]
                temp_score = temp_score / len(scores[level])
                print("class", level, "mean score :" ,temp_score)
            return scores
        else:
            return -1


    #TODO : generalize the repeated code below, this must appear at least 10 times in total and the only
    # thing that changes is when we call the inner function and setattr (which can probably be generalized instead of hard-coding)
    # Or just give what name to put in corpus-statistics as an argument. ezpz.
    # Sure it's hard coding but eh. EXTREMELY less bloat.
    def average_levenshtein_distance(self,mode = "old20"):
        """
        Returns the average Orthographic Levenshtein Distance 20 (OLD20), or its phonemic equivalent (PLD20).
        Currently using the Lexique 3.0 database for French texts, version 3.83. More details here : http://www.lexique.org/
        OLD20 is an alternative to the orthographical neighbourhood index that has been shown to correlate with text difficulty.

        :param str type: What kind of value to return, OLD20 or PLD20.
        :return: Average of OLD20 or PLD20 for each word in current text
        :rtype: Union[float,dict[str][list(float)]]
        """
        # Times for corpus tokens_split : [103.75190171699978, 476.5114750009998, 900.8312998260008, 1240.3407240750003]
        # TODO : Optimize this, probably by only calling import_lexique_dataframe() once.
        # Might need to make a function called average_levenshtein_distance_corpus() if it's worth the time gain
        func = word_list_based.average_levenshtein_distance
        if self.content_type == "text":
            return func(self.content,self.nlp,mode)
        elif self.content_type == "corpus":
            scores = {}
            for level in self.classes:
                temp_score = 0
                scores[level] = []
                for index,text in enumerate(self.content[level]):
                    scores[level].append(func(text, self.nlp, mode))
                    temp_score += scores[level][index]
                    if hasattr(self, "corpus_statistics"):
                        setattr(self.corpus_statistics[level][index],locals()['mode'],scores[level][index])
                temp_score = temp_score / len(scores[level])
                print("class", level, "mean score :" ,temp_score)
            return scores
        else:
            return -1

    # NOTE : These could be grouped together in the same function, and just set an argument type="X"
    def count_pronouns(self, mode="text"):
        func = discourse.nb_pronouns
        return func(self.content,self.nlp,mode)
    
    def count_articles(self, mode="text"):
        func = discourse.nb_articles
        return func(self.content,self.nlp,mode)
        
    def count_proper_nouns(self, mode="text"):
        func = discourse.nb_proper_nouns
        return func(self.content,self.nlp,mode)

    def lexical_cohesion_tfidf(self, mode="text"):
        func = discourse.average_cosine_similarity_tfidf
        return func(self.content,self.nlp,mode)

    # NOTE: this seem to output the same values, whether we use text or lemmas, probably due to the type of model used.
    def lexical_cohesion_LDA(self, mode="text"):
        func = discourse.average_cosine_similarity_LDA

        # ok so for our thing there are two things we need to pass
        # 1) the function alongside the needed arguments
        # 2) what name to give it for .corpus_statistics purposes. aka re-use for faster execution | use it in .scores() function eventually
        #return func(self.content,self.nlp,mode)

        return self.stub_generalize_get_score(func, (self.nlp,mode), "score_lda")


    def stub_rsrs():
        #TODO : check sobmodule for implementation details
        return -1


    def stub_SVM():
        #TODO: allow user to use default tf-idf matrix thing or with currently known features from other methods(common_scores text diversity, etc..)
        return -1

    def stub_MLP():
        #TODO: allow user to use default tf-idf matrix thing or with currently known features from other methods(common_scores text diversity, etc..)
        return -1

    def stub_compareModels():
        #NOTE: this is probably not too high on the priority list of things to do.
        return -1


    def stub_fastText():
        return -1
        
    def stub_BERT():
        return -1


    # Utility functions that I don't think I can put in utils
    def do_function_with_default_arguments(self,score_type):
        """Utility function that calls one of the function listed in self.score_types with default arguments"""
        if score_type in self.score_types["no_argument_needed"].keys():
            func = self.score_types["no_argument_needed"][score_type]
            print("WARNING: Score type '",score_type,"' not found in current instance, now using function ",func.__name__, " with default parameters",sep="")
            return func()
        elif score_type in self.score_types["argument_needed"].keys():
            func =  self.score_types["argument_needed"][score_type]
            print("WARNING: Score type '",score_type,"' not found in current instance, now using function ",func.__name__, " with default parameters",sep="")
            return func(score_type)
    
    def get_corpus_scores(self, score_type=None, scores=None):
        """
        Utility function that searches relevant scores if a valid score_type that belongs to self.score_types is specified.
        If no scores can be recovered, this function will attempt to call the corresponding function with default arguments.

        :return: a dictionary, assigning a value to each text of each specified class of a corpus.
        :rtype: dict[str][str][str][str]
        """
        # There are 5 cases:
        # score_type is unknown : error
        # score_type is known,scores are known : do the actual function
        # score_type is known, scores are unknown, but stored inside corpus.stats : extract them then do actual function
        # score_type is known, scores are unkown, can't be found inside corpus_stats : get them w/ default then do actual function
        # score_type is known, scores are unknown, and not stored : get them w/ default param then do actual function

        if self.content_type == "text":
            raise TypeError('Content type is not corpus, please load something else to use this function.')
        if score_type not in list(self.score_types['no_argument_needed'].keys()) + list(self.score_types['argument_needed'].keys()):
            raise RuntimeError("the score type '", score_type ,"' is not recognized by the current Readability object, please pick one from", list(self.score_types['no_argument_needed'].keys()) + list(self.score_types['argument_needed'].keys()))
        if score_type is not None:
            if not hasattr(self,"corpus_statistics"):
                print("Suggestion : Use Readability.compile() beforehand to allow the Readability object to store information when using other methods.")
                if scores is not None:
                    # Case : user provides scores even though .compile() was never used, trust these scores and perform next function.
                    print("Now using user-provided scores from 'scores' argument")
                    pass
                else:
                    # Case : user provides no scores and cannot refer to self to find the scores, so use corresponding function with default parameters.
                    scores = self.do_function_with_default_arguments(score_type)

            elif not hasattr(self.corpus_statistics[self.classes[0]][0],score_type):
                # Case : self.corpus_statistics exists, but scores aren't found when referring to self, so using corresponding function with default parameters.
                scores = self.do_function_with_default_arguments(score_type)
            else:
                # Case : scores found via referring to self.corpus_statistics, so just extract them.
                scores = {}
                for level in self.classes:
                    scores[level] = []
                    for score_dict in self.corpus_statistics[level]:
                        scores[level].append(score_dict.__dict__[score_type])

            return scores
    

    def remove_outliers(self, stddevratio=1, score_type=None, scores=None):
        """
        Outputs a corpus, after removing texts which are considered to be "outliers", based on a standard deviation ratio
        A text is an outlier if its value value is lower or higher than this : mean +- standard_deviation * ratio
        In order to exploit this new corpus, you'll need to make a new Readability instance.
        For instance : new_r = Readability(r.remove_outliers(r.perplexity(),1))

        :return: a corpus, in a specific format where texts are represented as lists of sentences, which are lists of words.
        :rtype: dict[str][str][str][str]
        """
        scores = self.get_corpus_scores(score_type,scores)
        moy = list()
        for level in self.classes:
            inner_moy=0
            for score in scores[level]:
                inner_moy+= score/len(scores[level])
            moy.append(inner_moy)

        stddev = list()
        for index, level in enumerate(self.classes):
            inner_stddev=0
            for score in scores[level]:
                inner_stddev += ((score-moy[index])**2)/len(scores[level])
            inner_stddev = math.sqrt(inner_stddev)
            stddev.append(inner_stddev)

        outliers_indices = scores.copy()
        for index, level in enumerate(self.classes):
            outliers_indices[level] = [idx for idx in range(len(scores[level])) if scores[level][idx] > moy[index] + (stddevratio * stddev[index]) or scores[level][idx] < moy[index] - (stddevratio * stddev[index])]
            print("nb textes enleves(",level,") :", len(outliers_indices[level]),sep="")
            print(outliers_indices[level])

        corpus_no_outliers = copy.deepcopy(scores)
        for level in self.classes:
            offset = 0
            for index in outliers_indices[level][:]:
                corpus_no_outliers[level].pop(index - offset)
                offset += 1
            print("New number of texts for class", level, ":", len(corpus_no_outliers[level]))
        print("In order to use this new corpus, you'll have to make a new Readability instance.")
        return corpus_no_outliers


    def stub_generalize_get_score(self, func, func_args , score_name):
        """
        Stub for functions that do the same logic when calculating a score and then potentially assign it to .corpus_statistics or else

        func is just a reference to the function
        func_args is a tuple  with the needed arguments for a text.
        score_name is the name of the score to assign to .corpus_statistics in order to re-use it in other methods.
        """
        if self.content_type == "text":
            print(func_args)
            print(*(func_args))
            return func(self.content, *(func_args))
        elif self.content_type == "corpus":
            scores = {}
            for level in self.classes:
                temp_score = 0
                scores[level] = []
                for index,text in enumerate(self.content[level]):
                    scores[level].append(func(text, *(func_args)))
                    temp_score += scores[level][index]
                    if hasattr(self, "corpus_statistics"):
                        setattr(self.corpus_statistics[level][index],score_name,scores[level][index])
                temp_score = temp_score / len(scores[level])
                print("class", level, "mean score :" ,temp_score)
            return scores
        else:
            return -1


    def compile(self):
        """
        Calculates a bunch of statistics to make some underlying functions faster.
        Returns a copy of a Readability instance, supplemented with a "statistics" or "corpus_statistics" attribute that can be used for other functions.
        """
        #TODO : debloat this and/or refactor it since we copy-paste almost the same below
        if self.content_type == "text":
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
        In this case, this will output the mean values of each score for each class.
        """
        if hasattr(self, "statistics"):
            for stat in self.statistics.__dict__:
                print(stat, "=", self.statistics.__dict__[stat])

        # NOTE: Might more more useful to instead return the mean values of each class in the corpus.
        elif hasattr(self, "corpus_statistics"):
            for level in self.classes:
                class_values = dict.fromkeys(self.corpus_statistics[level][0].__dict__, 0)
                for stats in self.corpus_statistics[level]:
                    for stat in stats.__dict__:
                        class_values[stat] += stats.__dict__[stat]
                print("Mean values for class :",level)
                class_values.update((key, round(value/len(self.corpus_statistics[level]),2)) for key,value in class_values.items())
                print(class_values)
        else:
            print("You need to use the .compile() function before in order to view this",self.content_type,"' statistics")


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
