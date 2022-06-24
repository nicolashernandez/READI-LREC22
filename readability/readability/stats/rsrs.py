"""
Don't know what to put here yet.
The RSRS feature is a novel idea developped by a 2019 Martinc paper : https://arxiv.org/abs/1907.11779
It is based on two ideas : Shallow lexical sophistication indicators work well, and can be used alongside values derived from a Language Model.
Currently, the perplexity score is an unweighted score even though there can be a small number of unreadable words in a text.
Therefore, assigning larger weights to such words might improve the correlation of LM values with readability :
Formula : text -> split into sentences -> WNLL (yt * log(yp) +(1 - yt) * log(1 - yp)) for each word
where yp is a probability derived from a language model according to historical sequence (don't use BERT),
and yt denotes the empirical distribution for a specific position in the sequence :
it's 1 for word that actually appears next in the sequence and 0 for everything else..
Then sort by ascending score and RSRS = (SUM root(rank_word) * WNLL(rank_word)) / sentence length
rank_word == index.
"""


# The RSRS feature based on two ideas :
# - Shallow lexical soph indicators, like length of sentence, usually correlate well with readi so they're used alongside stats derived from LM
# - Perplexity score is an unweighted sum, even though there are a small number of unreadable words.
# Assigning larger weights to such words might improve corr of LM scores with readability
# So, text -> split into sentences (default nltk) -> WNLL
# WNLL = (yt * log yp + (1 - yt) * log (1 - yp)), where yp = proba (softmax distrib) predicted by lm according to historical sequence
# # yt denotes empirical distrib for specific position in sentence, i.e has value 1 for word in vocab that actually appears next, and 0 for everything else.
# Then every word is sorted in ascensing WNLL order, and RSRS  is calculated :
# RSRS = (SUM root(rank_word) * WNLL(rank_word)) / sentence length
# 