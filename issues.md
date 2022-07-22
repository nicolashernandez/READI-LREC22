# Overview of issues:

First: This library is currently only usable with the french language. While spacy models can be used as a substitute for the "base" NLP processing, some functions depend on external resources that can't be easily reproduced. For instance, wordlist-based features would need to find wordlists that represent the same thing. That would also be the case for models containing only French data, like the one currently used in order to measure cosine similarity between sentences.

Second : An attempt was made to document the library. Docstrings in the reStructured Text are available next to each method and class, however no documents have been generated in order to have a proper documentation. Sphinx can probably be used.


## In readability.py:
> ⚠️ **FIXME:** Find a better way to handle the main NLP processor.

The current NLP processor, spacy, couldn't be located even after being added as a dependency in setup.cfg:  
fr_core_news_sm@https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.3.0/fr_core_news_sm-3.3.0.tar.gz#egg=fr_core_news_sm  
Therefore, some code is used when initializing the readability processor in order to download the spacy model if it cannot be found. There probably exists a better way to search the current user environment for if spacy is installed, and if the wanted model is available.

> ⚠️ **FIXME:** function **lexical_cohesion_LDA** might have an issue.

When calling this function, using parameter mode='lemma' instead of mode='text' often returns the same result. This may be normal behavior but it is suspicious.

> 📝 **TODO:** Rename function **score** to **traditional_score**.

In the module, the score function acts as a gateway to the submodule which calculates traditional scores. This can be fixed by renaming it to traditional_score across each file for clarification.

> 📝 **TODO:** Add associations for measures that can only be obtained from a collection of texts.

Both the *ParsedText* and *ParsedCollection* classes rely on the dictionary attribute **informations** from the *ReadabilityProcessor* in order to know which scores are available, and how to calculate them. However, there currently exists no equivalent for measures that can only be obtained from a collection of texts. For instance : the mean accuracy after a classification task using a Transformer model. While these measures can be manually obtained by calling the relevant functions from a *ParsedCollection* instance, it'd be useful to denote these, and their dependencies.

> 📝 **TODO:** Group together the functions **count_pronouns**, **count_articles**, **count_proper_nouns**.

These functions act the same way, and are based on the same subroutine, they should be grouped together to avoid duplicated code and potential errors.

> 📝 **TODO:** Finish listing functions that start with **count_type_mention_** or **count_type_opening_**.

As there are 11 possible values, 22 accessor functions must be created in order to link a score to the proper function. This is needed in order to have them show when calling the function **show_scores** from a *ParsedText* or *ParsedCollection* instance.

> 📝 **TODO:** When using a ML or DL function, allow user to use currently known features from other methods(common_scores, text diversity, etc..)

This implies passing information from a *ParsedText* or *ParsedCollection* instance into a parameter that defaults to None.


## In utils/utils.py:
> ⚠️ **FIXME:** Untested function **generate_corpus_from_folder**.

This function is supposed to generate a corpus, which can then be parsed into a *ParsedCollection*. However this function was never used during development and is currently untested.

> 📝 **TODO:** Improve function **syllablesplit**.

The current function is a poor estimator of the number of syllables in a word, as it only counts the number of vowels. An improvement has been proposed by only counting vowels that are not preceded by another vowel, but that is still not accurate enough to warrant a change.

> 📝 **TODO:** Add BERT and fasttext to the dependency system.

As of July 22, the models are loaded from within the `models` submodule which could potentially cause weird behavior.


## In stats/common_scores.py:
> ⚠️ **FIXME:** Several formulas are incorrect.

These being GFI, ARI due to using incorrect formulas, SMOG due to an error in calculating polysyllables, FRE due to a wrong variable assignation.
These were kept as is, in order to keep the original paper's experiments reproducible.


## In stats/discourse.py:
> ⚠️ **FIXME:** Recognizing personal pronouns isn't accurate enough.

Please fix in function **spacy_filter_coreference_count**.

> 📝 **TODO:** Figure out how to recognize deictic words.

Deictic words refer to a specific time, place, or person in context. Their semantic meaning is fixed, but the denoted meaning can change.
Example : I love *this* city.
Knowledge of the current location is needed to identify which city is being referred to.

> 📝 **TODO:** Implement other cohesion features.

There exists a notion called Lexical Tightness. Further research is needed to know what it is, and how to evaluate it.

> 📋 **NOTE:** Usage of coreference chains via coreferee library could be modified or improved.

Results don't seem very accurate, also coreferee only returns the head of the mention, so things like "New York" or "this woman" will be shortened to "New" and "this" respectively, which can be a problem later on.


## In stats/word_list_based.py:
> 📋 **NOTE:** Use of Dubois-Buyse word list could be improved by developping additional functions.

When given a text, we can show the amount of words that appear in the word-list, and additionally filter this list based on certain grades, but it could also be useful to show if these words appear often in certain grades or not. This could simply be done with a bar plot.


## In stats/syntactic.py:
> 📝 **TODO:** Develop this module, in order to evaluate grammatical complexity

Some feature ideas have been noted in the module's description.


## In stats/rsrs.py:
> 📝 **TODO:** Develop this module, introduces a novel idea for text complexity:

The source of this feature and a quick description expalining it is available in the module's description. It could probably be merged into the module stats/perplexity.py since it relies partly on it.


## In stats/diversity.py:
> 📝 **TODO:** Expand this module, some additional features can be developped.

These include: Yule's k, lexical density measures, and n-gram lexical features.


## In parsed_collection/parsed_collection.py:
> 📋 **NOTE:** Test use of parameter **iterable_arguments** in function **show_scores**

While in theory the current implementation allows to call a ReadabilityProcessor function with additional values for a text-by-text basis in order to speed up the process, there are still some doubts on whether this is the best way to proceed.


## For both parsed_text/parsed_text.py ***and*** parsed_collection/parsed_collection.py:
> 📝 **TODO:** Add behavior to function show_scores()

Currently, this function has only two possible behaviors by setting argument force to True or False: Either show only scores that have been calculated, or calculate every score before outputting. There should be a third which shows scores that have been already calculated *and* calculates the rest of the scores before outputting.