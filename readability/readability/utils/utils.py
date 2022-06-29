"""
The utils module contains common functions that are used by the other classes
or things that are useful in order to reproduce the contents of the READI paper.
"""
import pickle
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from unidecode import unidecode


# Note : remove this when we're done, this is just a quick dev workaround
# Will have to fix this with the new structure...
DATA_ENTRY_POINT = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../..', 'data'))
#print(os.path.abspath(os.path.dirname(__file__)))
def test_import(file_path):
    with open(os.path.join(DATA_ENTRY_POINT,file_path+".pkl"),"rb") as file:
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


def convert_text_to_string(text):
    if isinstance(text, str):
        doc = text

    elif any(isinstance(el, list) for el in text):
        doc = ''
        for sentence in text:
            doc = doc + ' ' + ' '.join(sentence)
        
    elif isinstance(text, list):
        doc = ' '.join(text)
    return doc

def convert_text_to_sentences(text,nlp):
    # Convert string to list(list(str))
    if isinstance(text, str):
        text = [[token.text for token in sent] for sent in nlp(text).sents]

    # Handling text that doesn't need to be converted
    elif any(isinstance(el, list) for el in text):
        pass

    # Handling text that was only converted into tokens (just list())
    elif isinstance(text, list):
        text = ' '.join(text)
        text= [[token.text for token in sent] for sent in nlp(text).sents]
    return text

def group_words_in_sentences(text):
    doc = []
    for sentence in text:
        doc.append(' '.join(sentence))
    return doc

def count_occurences_in_document(text, spacy_filter, nlp=None, mode="text"):
    """
    Returns the numbers of articles in a text, also available per sentence by giving the argument mode="sentence"

    :param text: Content of a text
    :type text: str or list(list(str))
    :param nlp: What natural language processor to use, currently only spacy is supported.
    :type nlp: spacy.lang
    :param string mode: What value to return, old20 or pld20.
    :return: Number of pronouns in a text, or average number of pronouns per sentence for this text
    :rtype: float
    """
    if mode == "sentence":
        text = convert_text_to_sentences(text, nlp)
        counter = 0
        for sentence in text:
            doc = nlp(' '.join(sentence))
            prepped_doc = spacy_filter(doc,nlp)
            counter += len(prepped_doc)
        return (counter / len(text))

    elif mode == "text":
        text = convert_text_to_string(text)
        doc = nlp(text)
        prepped_doc = spacy_filter(doc,nlp)
        return len(prepped_doc)

    raise TypeError("Type of parameter 'mode' cannot be '", type(mode),"', needs to be 'text', or 'sentence'")


def load_dependency(dependency_name):
    #TODO : go get every dependency import/configuration thing and return a dictionary of what's needed
    #It'll go in ReadabilityProcessor.dependencies[dependency] and can be accessed by other functions.
    if dependency_name == "GPT2_LM":
        model_name = "asi/gpt-fr-cased-small"
        # Load pre-trained model (weights)
        with torch.no_grad():
                model = GPT2LMHeadModel.from_pretrained(model_name)
                model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return dict(model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=100,
                    model_loaded = True)
    elif dependency_name == "dubois_dataframe":
        return dict(dummy_var="dummy_value")
    elif dependency_name == "lexique_dataframe":
        return dict(dummy_var="dummy_value")
    elif dependency_name == "BERT":
        return dict(dummy_var="dummy_value")
    elif dependency_name =="fastText":
        return dict(dummy_var="dummy_value")
    else:
        raise ValueError("Dependency '",dependency_name,"' was not recognized as a valid dependency.",sep="")

    return 0