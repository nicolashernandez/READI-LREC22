"""
The word_list_based module contains functions allowing to calculate various features, as long as they require a word list.

Currently, two of them are used : 
First : The Lexique 3.83 database, licence [CC BY SA 4.0] webpage (http://www.lexique.org/)
This is used to obtain the Orthographic Levenshtein Distance or the phonemic equivalent, of words in French.
This has been recognized as a cognitive feature by this paper :
INVESTIGATING READABILITY OF FRENCH AS A FOREIGN LANGUAGE WITH DEEP LEARNING AND
COGNITIVE AND PEDAGOGICAL FEATURES

The second resource is the Dubois-Buyse scale, a list of ~ 4000 words normally understood by 75% of children at a certain age,
graded by orthographic difficulty in the French school system.
This resource is free, and can be found on the net in various formats : We used this one as a convenient excel table format: (https://www.charivarialecole.fr/archives/1847)

Dubois-Buyse resource detail : File column names are : Mot / Echelon / Commentaire / Variantes / age / A.Scolaire / cycle
-  Mot contains the word itself
-  Echelon "ranks" the word between value 1 to 43 (higher value = knowledge of word acquired later in life)
-  Commentaire and Variantes are not useful, they indicate whether a word is composite, and contain the lemmatized form but this applies for only a few words out of several thousands
-  age and A.scolaire indicate the age, and corresponding grade where a word is ranked. 
-  cycle indicates the placement of a word within the five three-year cycles, although the fifth cycle ends early at the first year.
"""
import os
from collections import Counter
from math import isnan
import pandas as pd

from ..utils import utils

DATA_ENTRY_POINT = utils.DATA_ENTRY_POINT

###Using the Lexique 3.83 database###
def slim_lexique(df):
    """Not meant to be used by the readability processor: Reduces the size of the Lexique 3.0 database in order to keep only important columns"""
    # Remove duplicate values : down to 45k words
    df = df.drop_duplicates(subset="ortho")
    # Keep only relevant columns for now : From 35 columns to 3.
    df = df[['ortho','old20','pld20']]
    DATA_PATH = os.path.join(DATA_ENTRY_POINT,'lexique','Lexique383_slim.tsv')
    df.to_csv(DATA_PATH,sep = "\t", index=False)
    print("saved slimmer lexique in data/lexique/")
    return 0

def average_levenshtein_distance(df, text, nlp=None, mode = "old20"):
    """
    Returns the average Orthographic Levenshtein Distance 20 (old20), or its phonemic equivalent (pld20).
    Currently using the Lexique 3.0 database for French texts, version 3.83. More details here : http://www.lexique.org/
    These alternatives to the orthographical neighbourhood index have been shown to correlate with text difficulty,
    due to being related to the perceptual ambiguity of word recognition when there exists close orthographic neighbours.

    :param text: Content of a text
    :type text: str or list(list(str))
    :param nlp: What natural language processor to use, currently only spacy is supported.
    :type nlp: spacy.lang
    :param string mode: What value to return, old20 or pld20.
    :return: Average of old20 or pld20 for each word in current text
    :rtype: float
    """
    # Converting each recognized noun in text to lowercase, in order to check if there's a match.
    # Not using lemmas since we don't want to modify a word's suffixes or prefixes.
    text = utils.convert_text_to_string(text)
    prepped_text = [token.text.lower() for token in nlp(text) if (not token.is_punct)]
    
    # Get each word and count the number of times it appears
    counter = Counter(prepped_text)
    average_value = 0
    recognized_word_count = len(counter)

    if mode == "old20":
        for element in counter:
            # Check if element is contained in list before calculating old20.
            if element in df['ortho'].values:
                element_index = df['ortho'][df['ortho'] == element].index[0]
                average_value += (counter[element] * df['old20'][element_index])
            else:
                #print("WARNING: Skipping word", element, "appearing", counter[element], "times because it isn't recognized in the lexique database")
                recognized_word_count -= 1
    elif mode == "pld20":
        for element in counter:
            # Check if element is contained in list before calculating pld20.
            if element in df['ortho'].values:
                element_index = df['ortho'][df['ortho'] == element].index[0]
                # Additional check, sometimes pld20 doesn't contain a value: Lower word count by one
                if not isnan(df['pld20'][element_index]):
                    average_value += (counter[element] * df['pld20'][element_index])
                else: 
                    #print("WARNING: Skipping word", element, "appearing", counter[element], "times since it has no pld20 value")
                    recognized_word_count -= 1
            else:
                #print("WARNING: Skipping word", element, "appearing", counter[element], "times because it isn't recognized in the lexique database")
                recognized_word_count -= 1
    else:
        raise ValueError("Parameter 'type' with value", mode, "is not supported, please provide one of the following values instead : 'old20', 'pld20'")

    return average_value / recognized_word_count

###Using the Dubois-Buyse word list###
def dubois_buyse_ratio(df, text, nlp = None, typ = "total", filter = None):
    """
    Returns the proportion of words included in the Dubois-Buyse word list for a specific text

    This function can specify the ratio for specific echelons, ages|school grades, or three-year cycles:
    -  Mot contains the word itself
    -  Echelon "ranks" the word between value 1 to 43 (higher value = knowledge of word acquired later in life)
    -  Commentaire and Variantes are not useful, they indicate whether a word is composite, and contain the lemmatized form but this applies for only a few words out of several thousands
    -  age and A.scolaire indicate the age, and corresponding grade where a word is ranked. 
    -  cycle indicates the placement of a word within the five three-year cycles, although the fifth cycle ends early at the first year.
    
    :param str typ: Which column to use to filter the dubois-buyse dataframe.
    :param filter: A number, or two numbers, used to subset the dubois-buyse dataframe.
    :type filter: Union[int, list] 
    :return: something
    :rtype: float
    """
    # NOTE : possible improvement, add a plot bool parameter in order to show distribution of ratios for a certain type over all possible values.
    # For example : for each text in a corpus, when gradually adding words that are learned at later ages,
    # is the rate of words recognized in the list linear, quadratic, or something else? Could be useful.
    text = utils.convert_text_to_string(text)

    # Converting each recognized noun in text to its lemma in lowercase, in order to check if there's a match
    prepped_text = [token.lemma_.lower() for token in nlp(text) if (not token.is_punct)]
    noun_counter = Counter(prepped_text)
    total_words = 0
    total_words_in_list = 0
    number_words_in_list = 0
    number_words = 0
    if typ != "total":
        if not isinstance(filter,(int,tuple,list)):
            raise TypeError("Type of parameter 'filter' cannot be '", type(filter),"', needs to be int, tuple, or list")
        # TODO : figure out a way to debloat this since the logic is the same, only some specific values change.
        if typ == "echelon":
            if isinstance(filter,int):
                if not (1 <= filter <= 43):
                    raise ValueError("Value of parameter 'filter' cannot be less than 1 or more than 43 when using parameter 'echelon'")
                df = df.loc[df['Echelon'] == filter]
            elif isinstance(filter,(tuple,list)):
                for value in filter:
                    if not(1 <= value <= 43):
                        raise ValueError("Value of parameter 'filter' cannot be less than 1 or more than 43 when using parameter 'echelon'")
                df = df.loc[df['Echelon'].between(filter[0],filter[1])]
        elif typ == "age":
            if isinstance(filter,int):
                if not (6 <= filter <= 15):
                    raise ValueError("Value of parameter 'filter' cannot be less than 6 or more than 15 when using parameter 'age'")
                df = df.loc[df['age'] == filter]
            elif isinstance(filter,(tuple,list)):
                for value in filter:
                    if not(6 <= value <= 15):
                        raise ValueError("Value of parameter 'filter' cannot be less than 6 or more than 15 when using parameter 'age'")
                df = df.loc[df['age'].between(filter[0],filter[1])]
        elif typ == "cycle":
            if isinstance(filter,int):
                if not (2 <= filter <= 5):
                    raise ValueError("Value of parameter 'filter' cannot be less than 2 or more than 5 when using parameter 'cycle'")
                df = df.loc[df['cycle'] == filter]
            elif isinstance(filter,(tuple,list)):
                for value in filter:
                    if not(2 <= value <= 5):
                        raise ValueError("Value of parameter 'filter' cannot be less than 2 or more than 5 when using parameter 'cycle'")
                df = df.loc[df['cycle'].between(filter[0],filter[1])]
        else:
            raise ValueError("Parameter 'type' with value", typ, "is not supported, please provide one of the following values instead : 'total', 'echelon', 'age', or 'cycle'")
    
    for element in noun_counter:
        total_words += noun_counter[element]
        #number_words += 1
        total_words_in_list += noun_counter[element] if element in df['Mot'].values else 0
        #number_words_in_list += 1 if element in df['Mot'].values else 0

    #print("number of different words in text = ", number_words)
    #print("total of words in text (according to spacy) = ", total_words)
    #print("number of different words in word list = ", number_words_in_list)
    #print("total of words in word list = ", total_words_in_list)
    return(total_words_in_list / total_words)
