"""
The RSRS feature is a novel idea introduced by Martinc's 2019 paper: https://doi.org/10.48550/arXiv.1907.11779

It is based on two ideas : Shallow lexical sophistication indicators work well, and can be used alongside values derived from a Language Model.
Currently, the perplexity score is an unweighted score even though there can be a small number of unreadable words in a text.
Therefore, assigning larger weights to such words might improve the correlation of LM values with readability :
Formula : text -> split into sentences -> word negative log-likelihood(WNLL) for each word: (yt * log(yp) +(1 - yt) * log(1 - yp))
where yp is a probability derived from a language model according to historical sequence,
and yt denotes the empirical distribution for a specific position in the sequence:
Value of 1 for word in vocabulary that actually appears next in the sequence and 0 for everything else.
Then sort by ascending score and RSRS = SUM(root(rank_word) * WNLL(rank_word)) / sentence length
"""