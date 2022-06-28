"""
The discourse module contains functions allowing to calculate notions related to text cohesion.
Text cohesion means how parts of a text are related with explicit formal grammatical ties.
The following features can be measured : co-reference | anaphoric chains, entity density and specific cohesion features (lexical too)
and POS-tag based cohesion measures.
"""
import math
from ..utils import utils
from collections import Counter

# Paper by A.Todirascu freely available here in case I need more details : https://hal.archives-ouvertes.fr/hal-01430554/document

# Discourse cohesion features (Martinc):
# Coherence ~~ is text more than collection of unrelated sentences
# Cohesion : text represented by explicit formal grammatical ties aka how are parts related to each other
# Cohesion => co-reference, anaphoric chains, entity density and cohesion features, lexical cohesion measures,
# and POS tag-based cohesion measures.
# Clarification : Entity cohesion ~~ relative freq of possible transtions between syntactic functions played by same entity in adjacent sentences
# Lexical cohesion ~ features like frequency of content word repetition (adjacent), Latent Semantic Analysis for similarity,
# Lexical Tightness for mean value of Positiv Normalized Pointwise Mutual Information for all pairs of content-word tokens in text
# POS tag-based is measuring the ratio of pronoun and article parts-of-speech

# Todirascu analyzed 65 discourse features, but found that they don't contribute much compared to traditional or simple formulas,
# Let's make available the 6 that were significant with semi-partial correlation.
# Number of pronouns per word | Number of personal pronouns
# Average word length of entity
# Object to None : distance between 2 consecutive mentions of same chain is larger than 1 sentence.
# First mention in chain being specific deictic pronoun (deictic = meaning depends on context, "here","next Tuesday","the thing there", etc.)
# I can't figure which one was the 6th best feature.

# Try to implement each value, and their not so significative variant (could be useful, implementable at the same time).
# Refer to todirascu 4.1 to view
# Since we'll later implement a semi-partial correlation evaluation.

# lexical cohesion measure = average cosine similarity between adjacent sentences in ONE document :
# Use TF-IDF : (words | lemma | sub(nouns, proper names, pronouns)-words | sub-lemmas)
# Use LDA as LSA equivalent : 
# get pre-trained model(s)/data here https://fauconnier.github.io/#data
# func exemple here https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt
# gensim/lda/cosine tutorials : https://github.com/nicolashernandez/teaching_nlp/blob/main/M2-CN-2021-22_03_repr%C3%A9sentation_des_textes_sac_de_mots.ipynb
# https://github.com/nicolashernandez/teaching_nlp/blob/main/M2-CN-2021-22_04_repr%C3%A9sentation_vectorielle_continue.ipynb

#Future for chains :
# https://github.com/boberle/cofr
# https://github.com/explosion/coreferee
# https://github.com/mehdi-mirzapour/French-CRS

# Use a tool called RefGen for French? (longo & todirascu 2010,2013), however it can't recognize resumptive anaphora, and can't do complex referents
# like groups or collections of objects. also it ignores adverbs, and only identifies simple cases of antecedance.

# It might be relevant to include ways to test the importance of these features for users of this lib, maybe just provide multiple linear regression
# and semi-partial correlation at the same time.

# Pos-tag based features : 
# TODO : make a wrapper | decorator to indicate that these do pretty much the same thing
def nb_pronouns(text, nlp = None, mode="text"):
    """Returns the numbers of pronouns in a text, also available per sentence by giving the argument mode='sentence'"""
    # Pronoun tags available here, https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
    spacy_pronoun_tags = ["PRON", "PRP", "PRP$", "WP", "WP$", "PDAT", "PDS", "PIAT", "PIDAT", "PIS", "PPER", "PPOSAT", "PPOSS", "PRELAT", "PRELS", "PRF", "PWAT", "PWAV", "PWS", "PN"]

    def spacy_filter(doc, nlp):
        return [token.text for token in nlp(doc) if (token.pos_ in spacy_pronoun_tags)]
    
    return utils.count_occurences_in_document(text, spacy_filter, nlp, mode)

def nb_articles(text, nlp=None, mode="text"):
    """Returns the numbers of articles in a text, also available per sentence by giving the argument mode='sentence'"""
    def spacy_filter(doc, nlp):
        return [token.text for token in nlp(doc) if (token.morph.get("PronType") == ["Art"])]
    return utils.count_occurences_in_document(text, spacy_filter, nlp, mode)

def nb_proper_nouns(text, nlp=None, mode="text"):
    """Returns the numbers of proper nouns in a text, also available per sentence by giving the argument mode='sentence'"""
    def spacy_filter(doc, nlp):
        return [token.text for token in nlp(doc) if (token.pos_ == "PROPN")]
    return utils.count_occurences_in_document(text, spacy_filter, nlp, mode)

# lexical cohesion measure = average cosine similarity between adjacent sentences in ONE document :
# Use TF-IDF : (words | lemma | sub(nouns, proper names, pronouns)-words | sub-lemmas)
# Use LDA as LSA equivalent : 
# get pre-trained model(s)/data here https://fauconnier.github.io/#data
# func exemple here https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt
# gensim/lda/cosine tutorials : https://github.com/nicolashernandez/teaching_nlp/blob/main/M2-CN-2021-22_03_repr%C3%A9sentation_des_textes_sac_de_mots.ipynb
# https://github.com/nicolashernandez/teaching_nlp/blob/main/M2-CN-2021-22_04_repr%C3%A9sentation_vectorielle_continue.ipynb

# Measures related to lexical cohesion :

def tfidf_thing():
    #Cosine similarity between similar sentences...
    #project them into word space with tfidf.. sure.
    # pure text | lemmas | sub-group text | sub-group lemma
    # so I need to figure out how to recognize nouns and proper names and pronouns (I do for nouns and pronouns, need to check for proper names.)
    return 0

def entity_density(text,nlp=None, mode="document"):
    """
    Entity density ~~ total|average number of all/unique entities in document
    Argument mode allows to provide a feature for entire document, or sentence per sentence
    """
    
    return 0


def average_word_length_per_entity(text,nlp = None,mode="document"):
    """
    This feature was significant in Todirascu's paper.
    Argument mode allows to provide a feature for entire document, or sentence per sentence
    """
    doc = utils.convert_text_to_string(text)

    # TODO: If no entities are recognized then raise exception
    # Also, spacy's NER module seems to not be that great for French, I should try out NLTK real quick to see if there's any significant difference

    #Oh wait it does, just not if the very first word refers to the subject. Weird but ok.
    for ent in nlp(doc).ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
        print(len(ent.text))
    #Then do a simple mean
    return 0    

def stub_entity_cohesion(text,nlp=None):
    """
    Need to figure out how this is calculated (Pitler and Nenkova 2008)
    """
    return 0



def stub_lexical_cohesion(text,nlp=None):
    """
    # Lexical cohesion ~ features like frequency of content word repetition (adjacent), Latent Semantic Analysis for similarity,
    # Lexical Tightness for mean value of Positiv Normalized Pointwise Mutual Information for all pairs of content-word tokens in text
    """
    return 0
#Re use part of dubois for some of todirascu's lexical cohesion notions



# NOTE : Might need to use something else for this, either NLTK or coreferee, by the same company that develops Spacy : https://github.com/explosion/coreferee
def stub_coreference(text,nlp=None):
    return 0

def anaphoric_chain(text,nlp=None):
    return 0



# The following features were significant in Todirascu's paper but I don't quite know what they are.
# This is syntactic transition type?
def distance_object_to_none(text,nlp = None):
    """
    # Object to None : distance between 2 consecutive mentions of same chain is larger than 1 sentence.
    """
    return 0

#might skip this one if spacy can't recognize chains, or find another tool
#Deictic are things like "en,y"? Not sure
def first_chain_is_deictic(text,nlp=None):
    return 0