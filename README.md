# READI-LREC22

The resources present in this repository are presented in the following paper. 

> Nicolas Hernandez, Tristan Faine and Nabil Oulbaz, Open corpora and toolkit for assessing text readability in French, [2nd Workshop on Tools and Resources for People with REAding DIfficulties (READI@LREC)](https://cental.uclouvain.be/readi2022/accepted.html), Marseille, France, June, 24th 2022

Our work was consolidated into a python library that can be used to reproduce the paper's content :

# Readability
The readability class allows to evaluate the readability of a text by using traditional features, this can be done for multiple languages and is extendable to corpora through the use of machine learning and deep learning techniques to help differentiate between different classes of texts, based on their estimated readability level.  
**Note:** If using a corpus, it is recommended to provide the following format : dict[class_name][text][sentence][word]  
Also, while the Readibility class can handle tokenization on its own, it is recommended to provide do it before, and provide a text as a list of lists of words.  
## Usage :
from readability import Readibility  
r = Readibility(text|corpus)  
r.scores()

put "open with colab " button here. 