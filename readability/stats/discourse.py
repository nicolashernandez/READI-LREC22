"""
The discourse module contains functions allowing to calculate notions related to text cohesion.
In this case, text cohesion means how a text is represented by explicit formal grammatical ties.
Aka how are parts of the text related to each other.
The following features can be measured : co-reference, anaphoric chains, entity density and specific cohesion features (lexical too)
and POS-tag based cohesion measures.
"""

# Paper freely available here in case I need more details : https://hal.archives-ouvertes.fr/hal-01430554/document

# Discourse cohesion features :
# Coherence ~~ is text more than collection if unrelated sentences
# Cohesion : text represented by explicit formal grammatical ties aka how are parts related to each other
# Cohesion => co-reference, anaphoric chains, entity density and cohesion features, lexical cohesion measures,
# and POS tag-based cohesion measures.
# Clarification : Entity cohesion ~~ relative freq of possible transtions between syntactic functions played by same entity in adjacent sentences
# Lexical cohesion ~ features like frequency of content word repetition (adjacent), Latent Semantic Analysis for similariyu,
# Lexical Tightness for mean value of Positiv Normalized Pointwise Mutual Information for all pairs of content-word tokens in text
# POS tag-based is measuring the ratio of pronoun and article parts-of-speech

# Todirascu analyzed 65 discourse features, but found that they don't contribute much compared to traditional or simple formulas,
# Let's make available the 6 that were significant with semi-partial correlation.
# Number of pronouns per word / Number of personal pronouns
# Average word length of entity
# Object to None : distance between 2 consecutive mentions of same chain is larger than 1 sentence.
# First mention of a chain being specific deictic pronoun : 4 (deictic = meaning depends on context, "here","next Tuesday","the thing there", etc.)
# I can't figure which one was the 6th best feature.

# Use a tool called RefGen for French? (longo & todirascu 2010,2013), however it can't recognize resumptive anaphora, and can't do complex referents
# like groups or collections of objects. also it ignores adverbs, and only identifies simple cases of antecedance.


# It might be relevant to include ways to test the importance of these futures for users of this lib, maybe just provide multiple linear regression
# and semi-partial correlation at the same time.

#I stopped at 2   Propositional Logic of https://www.nltk.org/book/ch10.html but I don't think it's the best thing to follow

def stub_coherence():
    return 0

def stub_cohesion():
    return 0