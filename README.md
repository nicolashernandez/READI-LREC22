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

## Sources:
TODO