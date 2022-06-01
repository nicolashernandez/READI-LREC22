import math
import pandas as pd
import numpy as np
import spacy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from unidecode import unidecode
import string

# Note : I tried adding the spacy model as a dependency in setup.cfg:
# fr_core_news_sm@https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.3.0/fr_core_news_sm-3.3.0.tar.gz#egg=fr_core_news_sm
# But I can't figure out how to use it, so this is a workaround.

try:
    nlp = spacy.load('fr_core_news_sm')
    print("DEBUG: Spacy model location (already installed) : ",nlp._path)
except OSError:
    print('Downloading spacy language model \n'
                "(Should happen only once)")
    from spacy.cli import download
    download('fr_core_news_sm')
    nlp = spacy.load('fr_core_news_sm')
    print("DEBUG: Spacy model location : ",nlp._path)

#Pseudo-docstring:
#The stats module is used by the Readability class in order to output 



def dataset_stats(corpus):
    """
    Output several basic statistics such as number of texts, sentences, or tokens, alongside size of the vocabulary.
        
    :param tokens: Dictionary of lists of sentences (represented as a list of tokens)
    :type tokens: dict[class][text][sentence][token]

    :return: a pandas dataframe 
    :rtype: pandas.core.frame.DataFrame
    """

    # Extract the classes from the dictionary's keys.
    levels = list(corpus.keys())
    cols = levels + ['total']

    # Build vocabulary
    vocab = dict()
    for level in levels:
        vocab[level] = list()
        for text in corpus[level]:
            unique = set()
            for sent in text:
                #for sent in text['content']:
                for token in sent:
                    unique.add(token)
                vocab[level].append(unique)
    
    # Number of texts, sentences, and tokens per level, and on the entire corpus
    nb_ph_moy= list()
    nb_ph = list()
    nb_files = list()
    nb_tokens = list()
    nb_tokens_moy = list()
    len_ph_moy = list()

    for level in levels:
        nb_txt = len(corpus[level])
        nb_files.append(nb_txt)
        nbr_ph=0
        nbr_ph_moy =0
        nbr_tokens =0
        nbr_tokens_moy =0
        len_phr=0
        len_phr_moy=0
        for text in corpus[level]:
            nbr_ph+=len(text)
            temp_nbr_ph = len(text)
            nbr_ph_moy+=len(text)/nb_txt
            for sent in text:
                nbr_tokens+=len(sent)
                nbr_tokens_moy+= len(sent)/nb_txt
                len_phr+=len(sent)
            len_phr_moy+=len_phr/temp_nbr_ph
            len_phr=0
        len_phr_moy = len_phr_moy/nb_txt
        nb_tokens.append(nbr_tokens)
        nb_tokens_moy.append(nbr_tokens_moy)
        nb_ph.append(nbr_ph)
        nb_ph_moy.append(nbr_ph_moy)
        len_ph_moy.append(len_phr_moy)
    nb_files_tot = sum(nb_files)
    nb_ph_tot = sum(nb_ph)
    nb_tokens_tot = sum(nb_tokens)
    nb_tokens_moy_tot = nb_tokens_tot/nb_files_tot
    nb_ph_moy_tot = nb_ph_tot/nb_files_tot
    len_ph_moy_tot = sum(len_ph_moy)/len(levels)
    nb_files.append(nb_files_tot)
    nb_ph.append(nb_ph_tot)
    nb_tokens.append(nb_tokens_tot)
    nb_tokens_moy.append(nb_tokens_moy_tot)
    nb_ph_moy.append(nb_ph_moy_tot)
    len_ph_moy.append(len_ph_moy_tot)

    # Vocabulary size per class
    taille_vocab =list()
    taille_vocab_moy=list()
    all_vocab =set()
    for level in levels:
        vocab_level = set()
        for text in vocab[level]:
            for token in text:
                all_vocab.add(token)
                vocab_level.add(token)
        taille_vocab.append(len(vocab_level))

    taille_vocab.append(len(all_vocab))

    # Mean size of vocabulary
    taille_vocab_moy = list()
    taille_moy_total = 0
    for level in levels:
        moy=0
        for text in vocab[level]:
            taille_moy_total+= len(text)/nb_files_tot
            moy+=len(text)/len(vocab[level])
        taille_vocab_moy.append(moy)
    taille_vocab_moy.append(taille_moy_total)  

    # The resulting dataframe can be used for statistical analysis.
    df = pd.DataFrame([nb_files,nb_ph,nb_ph_moy,len_ph_moy,nb_tokens,nb_tokens_moy,taille_vocab,taille_vocab_moy],columns=cols)
    df.index = ["Nombre de fichiers","Nombre de phrases total","Nombre de phrases moyen","Longueur moyenne de phrase","Nombre de tokens", "Nombre de token moyen","Taille du vocabulaire", "Taille moyenne du vocabulaire"]
    return round(df,0)

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

# Text "must" be a list of sentences, which are lists of words.
def GFI_score(text):
    """
    Output the Gunning Ford something index TODO find description online.
        
    :param tokens: Dictionary of lists of sentences (represented as a list of tokens)
    :type tokens: dict[class][text][sentence][token]

    :return: a pandas dataframe 
    :rtype: pandas.core.frame.DataFrame
    """
    #TODO: provide an alternative way to calculate the score depending on format
    #Oh, this is just the pre-formating thing, wow i really get what he means now.
    #Type sanity check : Convert string to list.
    if type(text) == str:
        text = [token.text for token in nlp(text) if (not token.is_punct)]
        #this... normally shouldn't modify the original text?
        #wait. does gfi even include punctuation...?
        #oh god nabil what have we done.

    print(text)
    totalWords = 0
    totalSentences = len(text)
    longWords = 0
    for sent in text:
        print(type(sent))
        totalWords += len(sent)
        longWords += len([token for token in sent if len(token)>6])
    score = 0.4*((totalWords/totalSentences) + 100*longWords/totalSentences)
    print(totalWords)
    print(totalSentences)
    print(longWords)
    return score

def ARI_score(text):
    totalWords = 0
    totalSentences = len(text)
    totalCharacters = 0
    for sent in text:
        totalWords += len(sent)
        totalCharacters += sum(len(token) for token in sent)
    score = 4.71*((totalCharacters/totalWords) + 0.5*totalWords/totalSentences)-21.43
    return score

def FRE_score(text):
    totalWords = 0
    totalSentences = len(text)
    totalSyllables = 0
    for sent in text:
        totalWords += len(sent)
        totalSyllables += sum(syllablesplit(word) for word in sent)
    score_FRE = 206.835-1.015*(totalWords/totalSentences)-84.6*(totalSyllables/totalWords)
    return(score_FRE)

def FKGL_score(text):
    totalWords = 0
    totalSentences = len(text)
    totalSyllables = 0
    for sent in text:
        totalWords += len(sent)
        totalSyllables += sum(syllablesplit(word) for word in sent)
    score_FKGL = 0.39*(totalWords/totalSentences)+11.8*(totalSyllables/totalWords)-15.59
    return(score_FKGL)

# Note : need to tell Mr Hernandez that polysyllables erroneously returned
# their own number of syllables instead of just 1.
def SMOG_score(text):
    totalSentences = len(text)
    nbPolysyllables = 0
    for sent in text:
        nbPolysyllables += sum(1 for word in sent if syllablesplit(word)>=3) 
    score_SMOG = 1.043*math.sqrt(nbPolysyllables*(30/totalSentences))+3.1291
    return(score_SMOG)

def REL_score(text):
    totalWords = 0
    totalSentences = len(text)
    totalSyllables = 0
    for sent in text:
        totalWords += len(sent)
        totalSyllables += sum(syllablesplit(word) for word in sent)
    score_REL = 207-1.015*(totalWords/totalSentences)-73.6*(totalSyllables/totalWords)
    return(score_REL)
# Note :     
# I should add a check for if type(text) == str or list. (to perform tokenization or not)
# I suppose that wouldn't be disruptive behaviour.


#TODO : reformat this in order to have less clutter..
#Note : The following 2 functions will have to be redone once we change our project's structure
#the thing where we have r = readability class
#and we can do r.compile(text)
#in order to have a bunch of information already known, which makes life much more easier and optimized.
def traditional_scores(corpus):
    """
    Outputs a pandas dataframe containing the mean scores for various traditional readability measures.
    :param corpus: Dictionary of lists of sentences (represented as a list of tokens)
    :type corpus: dict[class][text][sentence][token]

    :return: a pandas dataframe 
    :rtype: pandas.core.frame.DataFrame
    """
    from scipy.stats import pearsonr
    import time
    t0 = time.perf_counter()
    levels = list(corpus.keys())
    GFI = {}
    ARI = {}
    FRE = {}
    FKGL = {}
    SMOG = {}
    REL = {}
    for level in levels:
        GFI[level] = []
        ARI[level] = []
        FRE[level] = []
        FKGL[level] = []
        SMOG[level] = []
        REL[level] = []
        for text in corpus[level]:
            GFI[level].append(GFI_score(text))
            ARI[level].append(ARI_score(text))
            FRE[level].append(FRE_score(text))
            FKGL[level].append(FKGL_score(text))
            SMOG[level].append(SMOG_score(text))
            REL[level].append(REL_score(text))
    
    # Calculating means
    moy_GFI = []
    moy_ARI = []
    moy_FRE = []
    moy_FKGL = []
    moy_SMOG = []
    moy_REL = []
    for level in levels:
        moy = 0
        for score in GFI[level]:
            moy+= score/len(GFI[level])
        moy_GFI.append(moy) ; moy = 0
        for score in ARI[level]:
            moy+= score/len(ARI[level])
        moy_ARI.append(moy)
        for score in FRE[level]:
            moy+= score/len(FRE[level])
        moy_FRE.append(moy) ; moy = 0
        for score in FKGL[level]:
            moy+= score/len(FKGL[level])
        moy_FKGL.append(moy) ; moy = 0
        for score in SMOG[level]:
            moy+= score/len(SMOG[level])
        moy_SMOG.append(moy) ; moy = 0    
        for score in REL[level]:
            moy+= score/len(REL[level])
        moy_REL.append(moy) ; moy = 0

    # Calculating Standard Deviation
    stddev_GFI = []
    stddev_ARI = []
    stddev_FRE = []
    stddev_FKGL = []
    stddev_SMOG = []
    stddev_REL = []
    for index, level in enumerate(levels):
        stddev=0
        for score in GFI[level]:
            stddev += ((score-moy_GFI[index])**2)/len(GFI[level])
        stddev_GFI.append(math.sqrt(stddev)) ; stddev = 0
        for score in ARI[level]:
            stddev += ((score-moy_ARI[index])**2)/len(ARI[level])
        stddev_ARI.append(math.sqrt(stddev)) ; stddev = 0
        for score in FRE[level]:
            stddev += ((score-moy_FRE[index])**2)/len(FRE[level])
        stddev_FRE.append(math.sqrt(stddev)) ; stddev = 0
        for score in FKGL[level]:
            stddev += ((score-moy_FKGL[index])**2)/len(FKGL[level])
        stddev_FKGL.append(math.sqrt(stddev)) ; stddev = 0
        for score in SMOG[level]:
            stddev += ((score-moy_SMOG[index])**2)/len(SMOG[level])
        stddev_SMOG.append(math.sqrt(stddev)) ; stddev = 0
        for score in REL[level]:
            stddev += ((score-moy_REL[index])**2)/len(REL[level])
        stddev_REL.append(math.sqrt(stddev)) ; stddev = 0

    # Calculating Pearson correlation
    pearson = []
    labels = []
    GFI_list = []
    ARI_list = []
    FRE_list = []
    FKGL_list = []
    SMOG_list = []
    REL_list = []
    for level in levels:
        for val in GFI[level]:
            GFI_list.append(val)
            labels.append(levels.index(level))
        for val in ARI[level]:
            ARI_list.append(val)
        for val in FRE[level]:
            FRE_list.append(val)
        for val in FKGL[level]:
            FKGL_list.append(val)
        for val in SMOG[level]:
            SMOG_list.append(val)
        for val in REL[level]:
            REL_list.append(val)

    maxGFI = max(GFI_list)
    GFI_list = [val/maxGFI for val in GFI_list]
    maxARI = max(ARI_list)
    ARI_list = [val/maxARI for val in ARI_list]
    maxFRE = max(FRE_list)
    FRE_list = [val/maxFRE for val in FRE_list]
    maxFKGL = max(FKGL_list)
    FKGL_list = [val/maxFKGL for val in FKGL_list]
    maxSMOG = max(SMOG_list)
    SMOG_list = [val/maxSMOG for val in SMOG_list]
    maxREL = max(REL_list)
    #FRE_list = [val/maxFRE for val in FRE_list]
    # ^In the notebook, this is incorrect, need to fix
    REL_list = [val/maxREL for val in REL_list]
    pearson.append(pearsonr(GFI_list,labels)[0])
    pearson.append(pearsonr(ARI_list,labels)[0])
    pearson.append(pearsonr(FRE_list,labels)[0])
    pearson.append(pearsonr(FKGL_list,labels)[0])
    pearson.append(pearsonr(SMOG_list,labels)[0])
    pearson.append(pearsonr(REL_list,labels)[0])

    math_formulas = pd.DataFrame([moy_GFI,moy_ARI,moy_FRE,moy_FKGL,moy_SMOG,moy_REL],columns=levels)

    math_formulas.index = ["The Gunning fog index GFI", "The Automated readability index ARI","The Flesch reading ease FRE","The Flesch-Kincaid grade level FKGL","The Simple Measure of Gobbledygook SMOG","Reading Ease Level"]
    math_formulas['Pearson Score'] = pearson
    math_formulas.columns.name = "Mean values"
    print(math_formulas)

    math_formulas_stddev = pd.DataFrame([stddev_GFI,stddev_ARI,stddev_FRE,stddev_FKGL,stddev_SMOG,stddev_REL],columns=levels)
    math_formulas_stddev.index = ["The Gunning fog index GFI", "The Automated readability index ARI","The Flesch reading ease FRE","The Flesch-Kincaid grade level FKGL","The Simple Measure of Gobbledygook SMOG","Reading Ease Level"]
    math_formulas_stddev.columns.name = "Std Dev"
    print(math_formulas_stddev)

    print("time elapsed perf counter:", time.perf_counter() - t0)
    return math_formulas

def traditional_scores_optimized(corpus):  
    """
    Outputs a pandas dataframe containing the mean scores for various traditional readability measures.
    :param corpus: Dictionary of lists of sentences (represented as a list of tokens)
    :type corpus: dict[class][text][sentence][token]

    :return: a pandas dataframe 
    :rtype: pandas.core.frame.DataFrame
    """
    #Optimization : Calculate each score at once, since they share some parameters.
    from scipy.stats import pearsonr
    import time
    t0 = time.perf_counter()
    levels = list(corpus.keys())
    GFI = {}
    ARI = {}
    FRE = {}
    FKGL = {}
    SMOG = {}
    REL = {}
    for level in levels:
        GFI[level] = []
        ARI[level] = []
        FRE[level] = []
        FKGL[level] = []
        SMOG[level] = []
        REL[level] = []
        for text in corpus[level]:
            totalWords = 0
            nbLongWords = 0
            totalSentences = len(text)
            totalCharacters = 0
            totalSyllables = 0
            nbPolysyllables = 0
            for sent in text:
                totalWords += len(sent)
                nbLongWords += len([token for token in sent if len(token)>6])
                totalCharacters += sum(len(token) for token in sent)
                totalSyllables += sum(syllablesplit(word) for word in sent)
                nbPolysyllables += sum(1 for word in sent if syllablesplit(word)>=3) 
            GFI[level].append(0.4*((totalWords/totalSentences) + 100*nbLongWords/totalSentences))
            ARI[level].append(4.71*((totalCharacters/totalWords) + 0.5*totalWords/totalSentences)-21.43)
            FRE[level].append(206.835-1.015*(totalWords/totalSentences)-84.6*(totalSyllables/totalWords))
            FKGL[level].append(0.39*(totalWords/totalSentences)+11.8*(totalSyllables/totalWords)-15.59)
            SMOG[level].append(1.043*math.sqrt(nbPolysyllables*(30/totalSentences))+3.1291)
            REL[level].append(207-1.015*(totalWords/totalSentences)-73.6*(totalSyllables/totalWords))
    

    #TODO : keep improving from here.
    # Calculating means
    moy_GFI = []
    moy_ARI = []
    moy_FRE = []
    moy_FKGL = []
    moy_SMOG = []
    moy_REL = []
    for level in levels:
        moy = 0
        for score in GFI[level]:
            moy+= score/len(GFI[level])
        moy_GFI.append(moy) ; moy = 0
        for score in ARI[level]:
            moy+= score/len(ARI[level])
        moy_ARI.append(moy)
        for score in FRE[level]:
            moy+= score/len(FRE[level])
        moy_FRE.append(moy) ; moy = 0
        for score in FKGL[level]:
            moy+= score/len(FKGL[level])
        moy_FKGL.append(moy) ; moy = 0
        for score in SMOG[level]:
            moy+= score/len(SMOG[level])
        moy_SMOG.append(moy) ; moy = 0    
        for score in REL[level]:
            moy+= score/len(REL[level])
        moy_REL.append(moy) ; moy = 0

    # Calculating Standard Deviation
    stddev_GFI = []
    stddev_ARI = []
    stddev_FRE = []
    stddev_FKGL = []
    stddev_SMOG = []
    stddev_REL = []
    for index, level in enumerate(levels):
        stddev=0
        for score in GFI[level]:
            stddev += ((score-moy_GFI[index])**2)/len(GFI[level])
        stddev_GFI.append(math.sqrt(stddev)) ; stddev = 0
        for score in ARI[level]:
            stddev += ((score-moy_ARI[index])**2)/len(ARI[level])
        stddev_ARI.append(math.sqrt(stddev)) ; stddev = 0
        for score in FRE[level]:
            stddev += ((score-moy_FRE[index])**2)/len(FRE[level])
        stddev_FRE.append(math.sqrt(stddev)) ; stddev = 0
        for score in FKGL[level]:
            stddev += ((score-moy_FKGL[index])**2)/len(FKGL[level])
        stddev_FKGL.append(math.sqrt(stddev)) ; stddev = 0
        for score in SMOG[level]:
            stddev += ((score-moy_SMOG[index])**2)/len(SMOG[level])
        stddev_SMOG.append(math.sqrt(stddev)) ; stddev = 0
        for score in REL[level]:
            stddev += ((score-moy_REL[index])**2)/len(REL[level])
        stddev_REL.append(math.sqrt(stddev)) ; stddev = 0

    # Calculating Pearson correlation
    pearson = []
    labels = []
    GFI_list = []
    ARI_list = []
    FRE_list = []
    FKGL_list = []
    SMOG_list = []
    REL_list = []
    for level in levels:
        for val in GFI[level]:
            GFI_list.append(val)
            labels.append(levels.index(level))
        for val in ARI[level]:
            ARI_list.append(val)
        for val in FRE[level]:
            FRE_list.append(val)
        for val in FKGL[level]:
            FKGL_list.append(val)
        for val in SMOG[level]:
            SMOG_list.append(val)
        for val in REL[level]:
            REL_list.append(val)

    maxGFI = max(GFI_list)
    GFI_list = [val/maxGFI for val in GFI_list]
    maxARI = max(ARI_list)
    ARI_list = [val/maxARI for val in ARI_list]
    maxFRE = max(FRE_list)
    FRE_list = [val/maxFRE for val in FRE_list]
    maxFKGL = max(FKGL_list)
    FKGL_list = [val/maxFKGL for val in FKGL_list]
    maxSMOG = max(SMOG_list)
    SMOG_list = [val/maxSMOG for val in SMOG_list]
    maxREL = max(REL_list)
    #FRE_list = [val/maxFRE for val in FRE_list]
    # ^In the notebook, this is incorrect, need to fix
    REL_list = [val/maxREL for val in REL_list]
    pearson.append(pearsonr(GFI_list,labels)[0])
    pearson.append(pearsonr(ARI_list,labels)[0])
    pearson.append(pearsonr(FRE_list,labels)[0])
    pearson.append(pearsonr(FKGL_list,labels)[0])
    pearson.append(pearsonr(SMOG_list,labels)[0])
    pearson.append(pearsonr(REL_list,labels)[0])

    math_formulas = pd.DataFrame([moy_GFI,moy_ARI,moy_FRE,moy_FKGL,moy_SMOG,moy_REL],columns=levels)

    math_formulas.index = ["The Gunning fog index GFI", "The Automated readability index ARI","The Flesch reading ease FRE","The Flesch-Kincaid grade level FKGL","The Simple Measure of Gobbledygook SMOG","Reading Ease Level"]
    math_formulas['Pearson Score'] = pearson
    print(math_formulas)


    print("time elapsed perf counter:", time.perf_counter() - t0)
    return math_formulas
#this takes roughly half as much time.

###############################################################################

# The following measures are for text diversity:
def type_token_ratio(text):
    """
    Outputs three ratios : ttr and root ttr : number of lexical items / number of words
    The denominator is squared for root ttr, should be more robust as a measure
    :param text
    :type text: str

    :return: Root ttr ratio
    :rtype: float
    """
    from collections import Counter
    # Type check + handle punctuation.
    if type(text) == list:
        doc = ' '.join(text)
    else:
        doc = text
    doc = doc.translate(str.maketrans('', '', string.punctuation))

    nb_unique = len(Counter(doc.split()))
    nb_tokens = len(doc.split())
    print("TTR ratio = ",nb_unique,"/",nb_tokens,":",nb_unique/nb_tokens)
    print("(Returning) Root TTR ratio = ",nb_unique,"/",nb_tokens**2,":",nb_unique/nb_tokens**2)
    return(nb_unique/nb_tokens**2)

# The following methods use the spacy "fr_core_news_sm" model to recognize lexical items.
def noun_token_ratio(text):
    """
    Outputs variant of the type token ratio, using unique and total amount of nouns
    :param text
    :type text: str


    :return: Noun ratio
    :rtype: float
    """
    from collections import Counter
    if type(text) == list:
        doc = ' '.join(text)
    else:
        doc = text

    # Might need to remove stopwords via checking token.is_stop
    nouns = [token.text for token in nlp(doc) if (not token.is_punct and token.pos_ == "NOUN")]
    nb_unique = len(Counter(nouns))
    nb_tokens = len(nouns)

    print("NTR ratio = ",nb_unique,"/",nb_tokens,":",nb_unique/nb_tokens)
    print("(Returning) Root TTR ratio = ",nb_unique,"/",nb_tokens**2,":",nb_unique/nb_tokens**2)
    return(nb_unique/nb_tokens**2)
    

class PPPL_calculator:
    def load_model(self):
        #TODO: This is a 486MB model, we should find a way to keep it locally.

        model_name = "asi/gpt-fr-cased-small"
        # Load pre-trained model (weights)
        with torch.no_grad():
                self.model = GPT2LMHeadModel.from_pretrained(model_name)
                self.model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.max_length = 100

        print("Model online, you can now use .PPPL_score()")
        #Apparently I can pass name as a parameter in init
        return 0
    def gpt2_pppl_score(self,sentence):
        tokenize_input = self.tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input[:self.max_length]])
        loss=self.model(tensor_input, labels=tensor_input)[0]
        return np.exp(loss.detach().numpy())
    def PPPL_score_text(self,text):
        if type(text) == list:
            tex = ''
            for sent in text:
                tex +=' '.join(sent)
                calcul = self.gpt2_pppl_score(tex.strip())
                return calcul
        elif type(text) == string:
            calcul = self.gpt2_pppl_score(tex.strip())
            return calcul
        else:
            #return type error
            print("todo: return type error")
            return -1
    #maybe make just one function and change behavior according to type.
    def PPPL_score(self,corpus,save = False):
        levels = list(corpus.keys())
        perplex = dict()
        nb_tot = 0
        for level in levels:
            perplex[level] = []
            ppl = 0
            for text in corpus[level]:
                tex = ''
                for sent in text:
                    tex +=' '.join(sent)
                perplex[level].append(self.gpt2_pppl_score(tex.strip()))
        return perplex

        #with open('perplex_jll.pkl','wb') as file:
        #    pickle.dump(perplex,file)
    def remove_outliers(self,perplex,stddevratio = 1):
        levels = list(perplex.keys())
        moy_ppl= list()
        for level in levels:
            moy=0
            for score in perplex[level]:
                moy+= score/len(perplex[level])
            moy_ppl.append(moy)
        outliers_indices = perplex.copy()
        for index, level in enumerate(levels):
            outliers_indices[level] = [idx for idx in range(len(perplex[level])) if perplex[level][idx] > moy_ppl[index] + (stddevratio * stddev_ppl[index]) or perplex[level][idx] < moy_ppl[index] - (stddevratio * stddev_ppl[index])]
            print(outliers_indices[level])
            print("nb textes enleves(",level,"):", len(outliers_indices[level]))
        import copy
        corpus_no_outliers = copy.deepcopy(corpus)
        for level in levels:
            offset = 0
            for index in corpus_no_outliers[level][:]:
                corpus_no_outliers[level].pop(index - offset)
                offset += 1
            print("Number of texts for class", level, ":", len(corpus_no_outliers[level]))
        return corpus_no_outliers


pppl_calculator = PPPL_calculator()

# Todo : put a custom error message so that user remembers to do load_model()
# i'd put it in the __init__ but both cases are not optimal
# case one, if creation at import : takes too long for first start, weird behavior, not wanted
# case two, if no creation at import : other functions rely on the pppl, so if they don't have access to it, everything breaks.
# Not too important for now.
