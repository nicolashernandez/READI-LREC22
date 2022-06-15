"""Might need to make a "word_list_based" folder instead and differenciate between types of features."""



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