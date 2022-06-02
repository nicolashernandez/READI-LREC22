# Readability
The readability class allows to evaluate the readability of a text by using traditional features, this can be done for multiple languages and can be extended to corpora by providing a way to use machine learning and deep learning techniques to help differentiate between different texts, based on their estimated readability level.  
## Usage :
from readability import Readibility
r = Readibility(text) 
r.GFI_score()  
***Note:** While the Readibility class can handle tokenization on its own, it is recommended to provide a text as a list of lists of words.  
Also, corpora format must be the following : dict[class_name][text][sentence][word]

 


