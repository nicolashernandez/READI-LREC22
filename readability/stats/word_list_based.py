"""Might need to make a "word_list_based" folder instead and differenciate between types of features."""

import csv
import os
import pandas as pd
from .. import utils
from collections import Counter
# Cognitive features :
# 1. Proportion of abstract and concrete words :
# Les Voisions de Le Monde lexical database, can derive 3 vars, prop abstract, prop concrete, texte coverage of list
# can do same with our debuyse?
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


#Dubois-Buyse :
# File looks like this : mot / echelon / commentaire / variantes / age / année scolaire / cycle
# age et année scolaire sont la même valeur techniquement
# Cycles de 3 ans (normal)
# Ignorer commentaire
# Variante contient le mot répété, ou une version simple (s'absenter => absenter)
# Give proportion of words appearing, absolute and relative..?

#ok that's nice but this thing doesn't give me the echelon or the variant.
#Maybe I should just make a list of every word and add characteristics
#But then i can't simply look up all grade X...
#maybe i should make two lists or directly manipulate the dataframe?

dubois_df = None

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

# Nombre total de mots apparaissant dans la liste, peut-être pas vraiment utile comme statistique mais bon.
def proportion_total(text,type="total"):
    #Suppose text is in list(list()) format, and go make a "convert_text_to_lists" function in utils..

    df = import_dubois_dataframe()

    if type == "total":
        print("count every appearence")
        #Just make a collection object and check whether the individual values exist then sum altogether instead of checking token per token
        for sentence in text:
            for token in sentence:
                print(len(df[df['Mot'].contains(token)]))
                #^ not the way to proceed, it does cara not full token
    elif type == "unique":
        print("count only once")

    return 0


# Proportion based on echelons, most common echelon, etc...
def proportion_echelon(text):
    df = import_dubois_dataframe()
    
    return 0

#Proportion based on cycle, most common cycle etc...
def proportion_cycle(text):
    doc = utils.convert_text_to_string(text)
    df = import_dubois_dataframe()
    word_list = generate_dubois_word_list(df)
    #D'aprés tout les mots du text... donner la "distribution" selon le cycle? deja recuperons les 4 fréquences relatives.


    #rahh j'ai effacé le code qui aurait pu m'etre utile..
    #print(list(word_list.keys()))

    return 0

# Distribution of grade :

