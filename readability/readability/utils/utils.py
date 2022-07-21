"""
The utils module contains common functions that are used by the other classes
or things that are useful in order to reproduce the contents of the READI paper.
"""
import pickle
import os
import sys
import pandas as pd
import requests
import subprocess
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from gensim.models import KeyedVectors
from unidecode import unidecode
from ..parsed_collection import parsed_collection


DATA_ENTRY_POINT = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))


def load_pickle(file_path):
    """Loads a .pkl file containing serialized data from the data/ folder, usually a corpus of texts"""
    with open(os.path.join(DATA_ENTRY_POINT,file_path+".pkl"),"rb") as file:
        return pickle.load(file)

# TODO: test this and fix if necessary.
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

# TODO: improve this
def syllablesplit(input):
    """Estimates the number of syllables in a word, by counting the number of vowels."""
    nb_syllabes = 0
    syllables='aeiouy'
    for char in input:
        for syl in syllables:
            if syl == unidecode(char.lower()):
                nb_syllabes+=1
                break
    return nb_syllabes


# The following function provides a better estimator since it does not count vowels that are preceded by another vowel, but is unused as it is
# still not accurate enough.
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
    """When given a possible text as an input, this forcibly converts it to a single string
    
    Types of possible inputs :
    * Text is already a string, in this case, the function simply returns the text.
    * If the text is composed of sentences, which have been split into tokens : the function takes care of concatening tokens and sentences.
    * If the text is composed of sentences, that have not been split into tokens, the function takes care of concatening sentences.

    :return: A text, which has been converted into a single string.
    :rtype: str:
    """
    if isinstance(text, str):
        doc = text

    elif any(isinstance(el, list) for el in text):
        doc = ' '.join(text[0][:-1]) + text[0][-1] # Make first sentence not start with a whitespace, and remove whitespace between text and last punctuation mark.
        for sentence in text[1:]:
            doc = doc + ' ' + ' '.join(sentence[:-1] ) + sentence[-1] # Remove whitespace between text and last punctuation mark.
        
    elif isinstance(text, list):
        doc = ' '.join(text)
    return doc

def convert_text_to_sentences(text,nlp):
    """When given a possible text as an input, this forcibly converts it to sentences, further split into tokens.
    
    Types of possible inputs :
    * Text is already a string, in this case, the function splits the text into sentences, and tokenizes each of them.
    * If the text is composed of sentences, which have been split into tokens : the function simply returns the text
    * If the text is composed of sentences, that have not been split into tokens, the function tokenizes each sentence.

    :return: A text, which has been converted into a single string.
    :rtype: str:
    """
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

def convert_corpus_to_list(corpus):
    """
    Converts a dict[class][text][sentence][token] structure into two lists, one containing the texts, and the other containing their label.    
    
    :returns:
        - corpus_as_list (py:class:list(str)) - Lists each text as tokenized sentences.
        - labels (py:class:list(str)) - List of indexes indicates to which class a text belongs to.
    """
    corpus_as_list=list()
    labels = list()
    if isinstance(corpus, parsed_collection.ParsedCollection):
        for label in list(corpus.content.keys()):
            for parsed_text in corpus.content[label]:
                tex = []
                labels.append(list(corpus.content.keys()).index(label))
                for sentence in parsed_text.content:
                    tex.extend(sentence)
                corpus_as_list.append(tex)
    else:
        for level in corpus.keys():
            for text in corpus[level]:
                tex = []
                labels.append(list(corpus.keys()).index(level))
                for sent in text:
                    for token in sent:
                        tex.append(token.replace('\u200b',''))
                corpus_as_list.append(tex)
    return corpus_as_list, labels

def group_words_in_sentences(text):
    """Used for compatibility with other functions. Converts a text into a list of sentences, where words have been concatenated into a single string."""
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
    :param spacy_filter: A comprehension list containing a filtered text after applying a spacy model.
    :param string mode: Whether to count for the entire text, or to normalize by sentences.
    :return: Number of occurences of something in a text, or average number of something per sentence for this text.
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


def load_dependency(dependency_name, nlp_processor=None):
    """
    Used by the readability processor: Loads a resource locally via disk or downloads via the internet (storing locally if possible)
    
    Information to be returned can be of any type, but it is preferred to use a dictionary in order to associate names and values.
    These informations are meant to be stored in the readability processor as follows :
    ReadabilityProcessor.dependencies[language_model_example] = utils.load_dependency("language_model_name")
    """
    if dependency_name == "GPT2_LM":
        print("importing GPT2 model..")
        model_name = "asi/gpt-fr-cased-small"
        # Load pre-trained model (weights)
        with torch.no_grad():
                model = GPT2LMHeadModel.from_pretrained(model_name)
                model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("imported GPT2 model")
        return dict(model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=100,
                    model_loaded = True)

    elif dependency_name == "dubois_dataframe":
        print("importing dubois-buyse data as dataframe")
        DATA_PATH = os.path.join(DATA_ENTRY_POINT,'word_list','Dubois_Buysse.xlsx')
        df=pd.read_excel(DATA_PATH)
        print("dubois-buyse dataframe imported")
        return dict(dataframe=df)

    elif dependency_name == "lexique_dataframe":
        print("importing lexique data as dataframe")
        DATA_PATH = os.path.join(DATA_ENTRY_POINT,'lexique','Lexique383_slim.tsv')
        df=pd.read_csv(DATA_PATH, sep = '\t')
        print("lexique dataframe imported")
        return dict(dataframe=df)

    elif dependency_name == "fauconnier_model":
        try:
            with open(os.path.join(DATA_ENTRY_POINT,"corpus_fauconnier.bin"), "rb") as f:
                model = KeyedVectors.load_word2vec_format(os.path.join(DATA_ENTRY_POINT,"corpus_fauconnier.bin"), binary=True, unicode_errors="ignore")
                print("imported french word2vec model")
                return model
        except IOError:
            url = "https://embeddings.net/embeddings/frWac_no_postag_no_phrase_500_cbow_cut100.bin"
            print("WARNING : Acquiring french word2vec model remotely since model was not found locally.")
            response = requests.get(url)
            with open(os.path.join(DATA_ENTRY_POINT,"corpus_fauconnier.bin"), "wb") as f:
                f.write(response.content)
            model = KeyedVectors.load_word2vec_format(os.path.join(DATA_ENTRY_POINT,"corpus_fauconnier.bin"), binary=True, unicode_errors="ignore")
            return model
    elif dependency_name == "coreferee":
        # Calling python -m coreferee install fr to make sure we have access to model if needed
        subprocess.check_call([sys.executable, "-m", "coreferee", "install", "fr"])
        nlp_processor.add_pipe("coreferee")
        return dict(dummy_var="dummy_value")
    #These depend on actual data to be initialized, so i'll do it when I clean up BERT/fastText.
    elif dependency_name == "BERT":
        return dict(dummy_var="dummy_value")

    elif dependency_name =="fastText":
        return dict(dummy_var="dummy_value")

    else:
        raise ValueError("Dependency '",dependency_name,"' was not recognized as a valid dependency.")
