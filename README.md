# READI-LREC22

The resources present in this repository are presented in the following paper:

> Nicolas Hernandez, Tristan Faine and Nabil Oulbaz, Open corpora and toolkit for assessing text readability in French, [2nd Workshop on Tools and Resources for People with REAding DIfficulties (READI@LREC)](https://cental.uclouvain.be/readi2022/accepted.html), Marseille, France, June, 24th 2022


# Readability
Our work was consolidated into a python library that can be used to reproduce the paper's content:  

The readability module allows to evaluate the readability of a text by using traditional features. This can be done for multiple languages and is extendable to corpora through the use of machine learning and deep learning techniques to help differentiate between different classes of texts, based on their estimated readability level.  

**Note:** If you use your own corpus, it is recommended to provide the following format: `dict[class_name][text][sentence][word]` 
Also, while the Readibility module can handle tokenization on its own, it is recommended to provide do it before, and provide a text as a list of lists of words.

A reproduction of the contents of the paper is available here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolashernandez/READI-LREC22/blob/main/readi_reproduction.ipynb)  

## Usage:

    from readability import Readibility  
    r = Readibility(text|corpus)  
    r.scores()

