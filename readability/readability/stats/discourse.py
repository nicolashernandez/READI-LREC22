"""
The discourse module contains functions allowing to calculate notions related to text cohesion.
Text cohesion means how parts of a text are related with explicit formal grammatical ties.
The following features can be measured : co-reference | anaphoric chains, entity density and specific cohesion features (lexical too)
and POS-tag based cohesion measures.
"""
import math
from ..utils import utils
from collections import Counter
import pandas as pd
import os
import requests
import coreferee

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from gensim import corpora
from gensim.matutils import cossim

# Paper by A.Todirascu freely available here in case I need more details : https://hal.archives-ouvertes.fr/hal-01430554/document

# Discourse cohesion features (Martinc):
# Coherence ~~ is text more than collection of unrelated sentences
# Cohesion : text represented by explicit formal grammatical ties aka how are parts related to each other
# Cohesion => co-reference, anaphoric chains, entity density and cohesion features, lexical cohesion measures,
# and POS tag-based cohesion measures.
# Clarification : Entity cohesion ~~ relative freq of possible transitions between syntactic functions played by same entity in adjacent sentences
# Lexical cohesion ~ features like frequency of content word repetition (adjacent), Latent Semantic Analysis for similarity,
# Lexical Tightness for mean value of Positiv Normalized Pointwise Mutual Information for all pairs of content-word tokens in text
# POS tag-based is measuring the ratio of pronoun and article parts-of-speech

# Todirascu analyzed 65 discourse features, but found that they don't contribute much compared to traditional or simple formulas,
# Let's make available the 6 that were significant with semi-partial correlation.
# Number of pronouns per word | Number of personal pronouns : V except for personal pronouns
# Average word length of entity : X
# Object to None : distance between 2 consecutive mentions of same chain is larger than 1 sentence. : X
# First mention in chain being specific deictic pronoun (deictic = meaning depends on context, "here","next Tuesday","the thing there", etc.) : X
# I can't figure which one was the 6th best feature.

# Try to implement each value, and their not so significative variant(s) (could be useful, implementable at the same time).
# Refer to todirascu 4.1 to view
# Since we'll later implement a semi-partial correlation evaluation.

# Future for chains :
# https://github.com/boberle/cofr (Seems good but might be hard-ish to implement) (uses BERT-Multilingual as a dependency)
# https://github.com/explosion/coreferee (doesn't show "obvious" referents, easy to implement but might be a tad hard to extract some features)
# ^ Models are 20GB in size??? jeez. Well it does require conversion into word vectors after all
# https://github.com/mehdi-mirzapour/French-CRS (not usable at all, requires end-user to make a virtualenv and jump through hoops)





# Use a tool called RefGen for French? (longo & todirascu 2010,2013), however it can't recognize resumptive anaphora, and can't do complex referents
# like groups or collections of objects. also it ignores adverbs, and only identifies simple cases of antecedance.

# It might be relevant to include ways to test the importance of these features for users of this lib, maybe just provide multiple linear regression
# and semi-partial correlation at the same time.

#found this guy talking about coreference chains https://boberle.com/fr/projects/coreference-chains-in-research-articles/
# read this to understand how stuff works https://aclanthology.org/P19-1066.pdf Coreference Resolution with Entity Equalization

spacy_pronoun_tags = ["PRON", "PRP", "PRP$", "WP", "WP$", "PDAT", "PDS", "PIAT", "PIDAT", "PIS", "PPER", "PPOSAT", "PPOSS", "PRELAT", "PRELS", "PRF", "PWAT", "PWAV", "PWS", "PN"]
DATA_ENTRY_POINT = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../..', 'data'))

# Pos-tag based features : 
# TODO : make a wrapper | decorator to indicate that these do pretty much the same thing
def nb_pronouns(text, nlp = None, mode="text"):
    """Returns the numbers of pronouns in a text, also available per sentence by giving the argument mode='sentence'"""
    # Pronoun tags available here, https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
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
def average_cosine_similarity_tfidf(text, nlp = None, mode="text"):
    """
    Returns the average cosine similarity between adjacent sentences in a text.
    By using the 'mode' parameter, can use inflected forms of tokens or the corresponding lemmas, possibly filtering the text beforehand
    in order to keep only nouns, proper names, and pronouns.
    Valid values for mode are : 'text', 'lemma', 'subgroup_text', 'subgroup_lemma'.
    """
    # Group sentences together to be compatible with the tfidf_vectorizer and prepare tokens depending on selected mode :
    # text | lemmas | sub-grouped text | sub-grouped lemmas
    sentences = utils.convert_text_to_sentences(text, nlp)
    prepped_text = []

    if mode == "text":
        def spacy_filter(doc, nlp):
            return [token.text for token in nlp(doc)]
    elif mode == "lemma":
        def spacy_filter(doc, nlp):
            return [token.lemma_ for token in nlp(doc)]
    elif mode == "subgroup_text":
        def spacy_filter(doc, nlp):
            return [token.text for token in nlp(doc) if (token.pos_ in spacy_pronoun_tags) or (token.pos_ == "PROPN") or (token.pos_ == "NOUN")]
    elif mode == "subgroup_lemma":
        def spacy_filter(doc, nlp):
            return [token.lemma_ for token in nlp(doc) if (token.pos_ in spacy_pronoun_tags) or (token.pos_ == "PROPN") or (token.pos_ == "NOUN")]
    else:
        raise TypeError("Type of parameter 'mode' cannot be '", mode,"', needs to be 'text', 'lemma', 'subgroup_text', 'subgroup_lemma'")
    
    for sentence_tokens in sentences:
        doc = ' '.join(sentence_tokens)
        prepped_text.append(spacy_filter(doc,nlp))

    doc = utils.group_words_in_sentences(prepped_text)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    average_cosine_similarity = 0

    # Average the cosine_similarity value between each adjacent sentence
    for index, submatrix in enumerate(similarity_matrix[:-1]):
        average_cosine_similarity += submatrix[index+1] 
    average_cosine_similarity = average_cosine_similarity / len(similarity_matrix[:-1])

    return average_cosine_similarity

def average_cosine_similarity_LDA(model, text, nlp = None, mode="text"):
    """
    Returns the average cosine similarity between adjacent sentences in a text.
    By using the 'mode' parameter, can use inflected forms of words or their lemmas.
    Valid values for mode are : 'text', 'lemma'.
    """
    # Group sentences together and keep text or lemmas
    sentences = utils.convert_text_to_sentences(text, nlp)
    prepped_text = []

    if mode == "text":
        def spacy_filter(doc, nlp):
            return [token.text for token in nlp(doc)]
    elif mode == "lemma":
        def spacy_filter(doc, nlp):
            return [token.lemma_ for token in nlp(doc)]
    else:
        raise TypeError("Type of parameter 'mode' cannot be '", type(mode),"', needs to be 'text', 'lemma'")
    
    for sentence_tokens in sentences:
        doc = ' '.join(sentence_tokens)
        prepped_text.append(spacy_filter(doc,nlp))

    # Convert sentences into BOW vectors
    dictionary = corpora.Dictionary(prepped_text)
    text_vectors = []
    for sentence in prepped_text:
        text_vectors.append(dictionary.doc2bow(sentence))
    #similarity = model.wmdistance(text_vectors[0], text_vectors[1])

    # Average the cosine_similarity value between each adjacent sentence
    average_cosine_similarity = 0
    # TODO: Handle texts that only have one sentence in them somehow.
    if len(text_vectors[:-1]) > 0:
        for index in range(len(text_vectors[:-1])):
            average_cosine_similarity = cossim(text_vectors[index], text_vectors[index+1])
        average_cosine_similarity = average_cosine_similarity / len(text_vectors[:-1])
    else:
        average_cosine_similarity = 0

    return average_cosine_similarity

# Things that can be done with coreferee :
# First off : need to downgrade spacy and the model from 3.3.0 to 3.2.0 due to compatibility issues. Oh well.
# Next, how to use :
# put coreferee in setup.cfg
# do python3 -m coreferee install fr (figure out how to do that from within script or within setup.cfg or elsewhere)

# nlp = spacy.load("fr_core_news_sm")
# nlp.add_pipe("coreferee")

# doc = nlp("Même si elle était très occupée par son travail, Julie en avait marre. Alors, elle et son mari décidèrent qu'ils avaient besoin de vacances. Ils allèrent en Espagne car ils adoraient le pays)

# Can use doc._.coref_chains to get the following :

# What is returned
# doc.coref_chains is a chain holder, it's just a group of chains
# doc.coref_chains[0] is a chain, it holds mentions, can also do X.most_specific_mention_index to get most relevant representation of entity
# doc.coref_chains[0][0] is a mention, can hold multiple things in it (a composite mention like "Jane et son mari" would give [Jane,Mari])
# It gives a token index so we can just do doc[doc.coref_chains[0][0].token_indexes[0]] to access the token and do spacy stuff
# Of course this is a naive way that only works for mentions that refer to a singular entity.

# Important note : Using only this won't be able to get the features noted in 3. entity cohesion with the transitions 
# aka "subject to subject" where X is a subject in sentence n, and the reference to X in sentence n+1 is also a subject.
# these can take value subject, object, other, or none (when does not appear)
# So our developped method should consider sentence per sentence and try to figure out something.
# But we'll develop that after the "basic" use of the previous

# first n°32 and 33 : Number of entities per document normalized by number of words
# And proportion of unique entities normalized by number of words.

def entity_density(text,nlp=None,unique=False):
    """
    Entity density ~~ total|average number of all/unique entities in document
    :param bool unique: Whether to return proportion of all entities in document, or only unique entities.
    """
    doc = utils.convert_text_to_string(text)

    if unique:
        return len(nlp(doc)._.coref_chains) / len(text)
    else :
        counter = 0
        for chain in nlp(doc)._.coref_chains:
            counter += len(chain)
        return counter / len(text)

# And 34 : Average number of words per entity normalized by number of words.
# Problem is that coreferee only gives us one token per entity.. so things like New York will be shortened to New
# If only I could combine merge_entities with this.. or... maybe i call merge entities on this then i use coreferee?
# Nevermind, coreferee ignores merge_entities. 
# And using spacy.ents only works for named entities. hmmm..
# Perhaps I could get every index, check if one of them is the same as an ent.start in list of ents
# If so, then get full size
# However this won't work for not-named entities that are composite, like "cette femme".
def average_word_length_per_entity(text,nlp=None):
    """
    This feature was significant in Todirascu's paper.
    Argument mode allows to provide a feature for entire document, or sentence per sentence
    """
    doc = utils.convert_text_to_string(text)
    
    #Okay so get the

    
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
def first_chain_is_deictic(text,nlp=None):
    return 0