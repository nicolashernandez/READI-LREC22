"""
The utils module contains common functions that are used by the other classes
or things that are useful in order to reproduce the contents of the READI paper.
"""
import pickle
import os
from unidecode import unidecode


# Note : remove this when we're done, this is just a quick dev workaround
def test_import(file_path):
    with open("test_data/"+file_path+".pkl","rb") as file:
        return pickle.load(file)

# This returns a dictionary containing the content of each text in a dictionary :
# Note : I need to test this on different OS to make sure it works independently.
# If I remember correctly, it produces the following : dict[class][text]
# So we need to continue developping this. 
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

def syllablesplit(input):
    nb_syllabes = 0
    syllables='aeiouy'
    for char in input:
        for syl in syllables:
            if syl == unidecode(char):
                nb_syllabes+=1
                break
    return nb_syllabes
# ^ Current syllable splitter used in the notebooks (without the break)

#The following function provides a better estimator, but is unused as it is not accurate enough.
#def bettersyllablesplit(input):
#    nb_syllabes = 0
#    syllables='aeiouy'
#    prev_is_syl = False
#    for char in input:
#        if prev_is_syl:
#                prev_is_syl = False
#                continue
#        for syl in syllables:
#            if syl == unidecode(char) and not prev_is_syl:
#                nb_syllabes+=1
#                prev_is_syl = True
#                break
#    return(nb_syllabes)