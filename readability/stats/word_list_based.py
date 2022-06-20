"""Might need to make a "word_list_based" folder instead and differenciate between types of features."""

import os
import pandas as pd
from .. import utils
from collections import Counter
# Cognitive features :
# 1. Proportion of abstract and concrete words :
# Les Voisins de Le Monde lexical database, can derive 3 vars, prop abstract, prop concrete, texte coverage of list
# Goriachun, D. & N. Gala (2020). Identifying abstract and concrete words in French
# to better address reading difficulties. In E.L.R. Association (ed.) Proceedings
# of the 1st Workshop on Tools and Resources to Empower People with REAding
# DIfficulties (READI). Marseille, France, 33–40.

# 2. Avg OLD20(Orthographic Levenshtein Distance) / PLD20(Phonological Levenshtein distance) per word 
# = Average distance of 20 closest words found in lexicon, or closest phonemes.
# Can be found on the 125,623 entries of the Lexique 3.6 database.
# 3. Avg number of commonly known senses per word aka polysemy

# There apparently exists a list of 23,342 annotated French words, done by francois et al 2016
# Pedagogical features :
# variables designed from official Reference Level Descriptors for French (Beacco et al 2008)
# Ch 4 and 6, 5,518 entries from 8,486 (dropped duplicate)
# Features : prop of words associated to A1,A2,B1,B2 and what's not covered aka difficult
# Features based on Distributed Representations
# aka use the embeddings from deep learning.
# extract per sentence, then mean across for a passage.


# Dubois-Buyse information:
# File column names are : Mot / Echelon / Commentaire / Variantes / age / A.Scolaire / cycle
# Mot contains the word itself
# Echelon "ranks" the word between value 1 to 43 (higher value = knowledge of word acquired later in life)
# Commentaire and Variantes are not useful, they indicate whether a word is composite, and the lemmatized form but this applies for only a few words out of several thousands
# age and A.scol indicate the age, and corresponding grade where a word is ranked. 
# cycle indicates the placement of a word within the five three-year cycles, although the fifth one only contains the first year.

dubois_df = None
# Helper functions
def import_dubois_dataframe():
    global dubois_df
    if not isinstance(dubois_df,pd.DataFrame):
        print("dubois dataframe imported")
        DATA_PATH = os.path.join(os.getcwd(),'data','word_list','Dubois_Buysse.xlsx')
        df=pd.read_excel(DATA_PATH)
        dubois_df = df
        return df
    else:
        print("dubois dataframe already imported")
        return dubois_df

def generate_dubois_word_list(df):
    word_list = {}
    for cycle in df["A.Scolaire"].unique():
        word_list[cycle] = [x for x, y in zip(df['Mot'], df['A.Scolaire']) if y == cycle]
    return word_list

def dubois_proportion(text, nlp = None, typ = "total", filter = None):
    """
    Returns the proportion of nouns included in the Dubois-Buyse word list for a specific text
    This function can specify the ratio for specific echelons, ages|school grades, or three-year cycles
    """
    # TODO : allow filter to take the value "all" in order to print each ratio for a certain type(type= dataframe column, like age or echelon)
    # NOTE : possible improvement, add a plot bool parameter in order to show distribution of ratios for a certain type over all possible values.
    df = import_dubois_dataframe()
    text = utils.convert_text_to_string(text)

    # Converting each recognized noun in text to its lemma in lowercase, in order to check if there's a match
    prepped_text = [token.lemma_.lower() for token in nlp(text) if (not token.is_punct)]
    noun_counter = Counter(prepped_text)
    total_nouns = 0
    total_nouns_in_list = 0
    number_nouns_in_list = 0
    number_nouns = 0
    if typ == "echelon":
        if not(1 <= filter <= 43):
            raise ValueError("Value of parameter 'filter' cannot be less than 1 or more than 43 when using parameter 'echelon'")
        df = df.loc[df['Echelon'] == filter]
    elif typ == "age":
        if not(6 <= filter <= 15):
            raise ValueError("Value of parameter 'filter' cannot be less than 6 or more than 15 when using parameter 'age'")
        df = df.loc[df['age'] == filter]
    elif typ == "cycle":
        if not(2 <= filter <= 5):
            raise ValueError("Value of parameter 'filter' cannot be less than 2 or more than 5 when using parameter 'cycle'")
        df = df.loc[df['cycle'] == filter]
    else:
        raise ValueError("Parameter 'type' with value", typ, "is not supported, please provide one of the following values instead : 'total', 'echelon', 'age', or 'cycle'")
    
    for element in noun_counter:
        total_nouns += noun_counter[element]
        number_nouns += 1
        total_nouns_in_list += noun_counter[element] if element in df['Mot'].values else 0
        number_nouns_in_list += 1 if element in df['Mot'].values else 0

    #print("number of different nouns altogether = ", number_nouns)
    #print("total of nouns (according to spacy) = ", total_nouns)
    #print("number of different nouns in word list = ", number_nouns_in_list)
    #print("total of nouns in word list = ", total_nouns_in_list)
    return(total_nouns_in_list / total_nouns)
