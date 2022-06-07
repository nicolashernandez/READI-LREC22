# Readability
The readability class allows to evaluate the readability of a text by using traditional features, this can be done for multiple languages and is extendable to corpora through the use of machine learning and deep learning techniques to help differentiate between different classes of texts, based on their estimated readability level.  
**Note:** If using a corpus, format must be the following : dict[class_name][text][sentence][word]  
Also, while the Readibility class can handle tokenization on its own, it is recommended to provide do it before, and provide a text as a list of lists of words.  
## Usage :
from readability import Readibility  
r = Readibility(text|corpus)  
r.scores()
