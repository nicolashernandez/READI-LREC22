# the utils module contains functions that are used by most of the other classes
# or things that aren't necessary for the library.
import pickle
import os

# Note : remove this when we're done, this is just a quick workaround
def test_import(file_path):
    with open("test_data/"+file_path+".pkl","rb") as file:
        return pickle.load(file)

# This returns a dictionary containing the content of each text in a dictionary :
# corpus[nameoffolder][index]
def generate_corpus_from_folder(folder_path):
    #TODO: if no folder_path was specified, return error.
    corpus = dict()
    for top, dirs, files in os.walk(folder_path):  
        globals()[top.split(os.path.sep)[-1]] = list()
        for file in files:
            if file.endswith('txt'):
                with open(os.path.join(top,file),"r") as f:
                    text = f.read().replace('\n',' ').replace('  \x0c','. ')
            globals()[top.split(os.path.sep)[-1]].append(text)
        corpus[top.split(os.path.sep)[-1]] = globals()[top.split(os.path.sep)[-1]]

    corpus.pop("textFiles")
    # This only work if files are structured as follow : root/levelX/files
    return corpus

# Note : this does the same as the previous function, but should be more generalizable
def generate_corpus_from_folder_alt(folder_path):
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
