# READI-LREC22
The resources present in this repository are presented in the following paper:

> Nicolas Hernandez, Tristan Faine and Nabil Oulbaz, Open corpora and toolkit for assessing text readability in French, [2nd Workshop on Tools and Resources for People with REAding DIfficulties (READI@LREC)](https://cental.uclouvain.be/readi2022/accepted.html), Marseille, France, June, 24th 2022

# Readability
Our work was consolidated into a python library that can be used to reproduce the paper's content:  

The readability module allows to evaluate the readability of a text by using various features. This is currently only available for French, and is extendable to corpora through the use of machine learning and deep learning techniques to help differentiate between different classes of texts, based on their estimated readability level.

**Note:** If you use your own corpus, it is recommended to tokenize it beforehand to respect this format: `dict[class_name][text][sentence][word]`.  
However, it still recognizes the following formats : `dict[class_name][text]`, `list(list(text))`, or even `list(text)` for a simple collection of texts with no labels.

A reproduction of the contents of the paper is available here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolashernandez/READI-LREC22/blob/main/readi_reproduction.ipynb).  
It also contains an introduction to the library with some explanations on how to use it.

## Building the library:
Complying with PEP-517 and PEP-518, the files `pyproject.toml`, `setup.cfg`, and `MANIFEST.md` are used to build the library on your system.

`setup.cfg` is the most important one, it indicates the project's meta-information and build details.  
`MANIFEST.in` is used to keep files that are not essential parts of the build, such as external resources.  
`pyproject.toml` is used to configure the build system itself.

After cloning this git repository, go inside it and simply install the library by doing `pip install .`  
Then import from a python session: `import readability`

## Understanding the library:

### External resources:
Inside the `readability` folder, there exists a folder called `data`, At its root, there are files that contain the corpuses used in the aforementioned paper.  
Small-size external resources used by the library are kept in subfolders, but larger resources will be cached there when using the library.

### Library structure:
At the root of the library's folder, the first main component, the *ReadabilityProcessor* is located in the `readability.py` file.

The `stats` folder contains a bunch of python files containing the functions used to calculate measures, their name alludes to which notion they belong. For instance, `diversity.py` contains functions meant to calculate features related to text diversity, such as the text token ratio.  

The `methods` folder contains functions meant to help develop and use Machine Learning applications, such as the use of support vector machines or multi layer perceptrons as text classifiers.

The `models` folder contains functions meant to help develop and use Deep Learning applications, such as the use of fasttext or BERT language models as text classifiers.

Next, the `parsed_text` and `parsed_collection` folders contain the specification of the second main components, the *ParsedText* and *ParsedCollection* classes. These describe a bunch of accessor functions to interact with the readability processor, and also how to store and output results.

Finally, the `utils` folder contains diverse helper functions that can be used by most of the other submodules. It also contains a configuration of the external resources : How to acquire them, and what to extract from them.

## Using the library:
As mentioned before, the library constitues of two main components: the *ReadabilityProcessor*, and the *ParsedText* or *ParsedCollection* classes.  
First, the ReadabilityProcessor is created, and loads several external resources, cached locally or via the internet, which enables the several functions and implementations of readability estimation available.  
Then, the *ParsedText* or *ParsedCollection* instance should be created, in order to speed up the process via sharing information inside the *ReadabilityProcessor*, and to be able to quickly get the calculated measures wanted.

### Quick example of use:

    import readability
    readability_processor = readability.Readability(exclude=[...])
    readability_processor.informations.keys() # view the kinds of scores that can be calculated
    parsed_text = readability_processor.parse("example_text")
    parsed_corpus = readability_processor.parseCollection(example_corpus)
    parsed_corpus.show_scores(force=True)

## References:

### External resources:

#### Main NLP processor:

spacy [https://spacy.io/](https://spacy.io/)  
Coreferee pipeline plugin for coreference resolution [https://spacy.io/universe/project/coreferee](https://spacy.io/universe/project/coreferee)

#### Word lists:

Dubois-Buyse [https://www.charivarialecole.fr/archives/1847](https://www.charivarialecole.fr/archives/1847)  
Lexique database licence **[CC BY SA 4.0]** webpage [http://www.lexique.org/](http://www.lexique.org/)

#### Language models:

GPT2 model (used for perplexity) [https://huggingface.co/asi/gpt-fr-cased-small](https://huggingface.co/asi/gpt-fr-cased-small)  
French word2vec model by Jean-Philippe Fauconnier [https://fauconnier.github.io/#data](https://fauconnier.github.io/#data)  
fasttext [https://fasttext.cc/](https://fasttext.cc/)  
BERT description: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)  
BERT model source:[https://huggingface.co/docs/transformers/model_doc/bert](https://huggingface.co/docs/transformers/model_doc/bert)

### Source of implementations:
For text diversity, text token ratio had been mentioned in this paper: [ A large-scaled corpus for assessing text readability](https://link.springer.com/article/10.3758/s13428-022-01802-x)  
Text cohesion features were describe in this paper, although implementations are entirely original: [Are Cohesive Features Relevant for Text Readability Evaluation?](https://hal.archives-ouvertes.fr/hal-01430554)  
Details regarding pseudo-perplexity came from this paper: [Masked Language Model Scoring](https://doi.org/10.48550/arXiv.1910.14659)  
The Orthographic Levenshtein Distance 20 had been described in the paper(INVESTIGATING READABILITY OF FRENCH AS A FOREIGN LANGUAGE WITH DEEP LEARNING AND COGNITIVE AND PEDAGOGICAL FEATURES) by Kevin Yancey, Alice Pintard, and Thomas François. It will be available here after the embargo period is lifted [https://dial.uclouvain.be/pr/boreal/fr/object/boreal%3A255445/datastreams](https://dial.uclouvain.be/pr/boreal/fr/object/boreal%3A255445/datastreams)