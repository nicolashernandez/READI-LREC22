"""
The utils module contains common functions that are used by the other classes
or things that are useful in order to reproduce the contents of the READI paper.

It will probably go alongside the library
"""
import pickle
import os




# Note : remove this when we're done, this is just a quick dev workaround
def test_import(file_path):
    with open("test_data/"+file_path+".pkl","rb") as file:
        return pickle.load(file)

# This returns a dictionary containing the content of each text in a dictionary :
# Note : I need to test this on different OS to make sure it works independently.
def generate_corpus_from_folder(folder_path):
    """
    Creates a dictionary with the same structure as the one used in our READI paper
        
    :param folder_path: Dictionary of lists of sentences (represented as a list of tokens)
    :type folder_path: string

    :return: A dictionary in our READI corpus format : dict[class][sentence_index][word_index]
    :rtype: dict
    """
    corpus = dict()
    for top, dirs, files in os.walk(os.getcwd()):  
        if top.endswith(folder_path):
            globals()[top.split(os.path.sep)[-1]] = list()
            for file in files:
                if file.endswith('txt'):
                    with open(os.path.join(top,file),"r") as f:
                        text = f.read().replace('\n',' ').replace('  \x0c','. ')
                    if len(text)>0:    
                        globals()[top.split(os.path.sep)[-1]].append(text)
            corpus[top.split(os.path.sep)[-1]] = globals()[top.split(os.path.sep)[-1]]
    return corpus

def compile(text):
    """
    Creates a dictionary with the same structure as the one used in our READI paper
        
    :param text: Preferably a list of sentences, which are lists of texts, but could be a string.
    :type text: list(list()) OR str

    :return: A readability object
    :rtype: readib.readability
    """

    if type(text) == str:
        print("Type Sanity Check : do the conversion from string to list of lists for later use")

    print("Calculate a bunch of useful information")

    print("return a readability class object, but with the extra information")

    return 0