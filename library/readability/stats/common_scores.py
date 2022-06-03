"""
The common_scores module contains functions allowing to calculate GFI, ARI, FRE, FKGL, SMOG, and REL.
Can be improved by changing formulas/calculations depending on language.
"""
import math

from unidecode import unidecode
import pandas as pd

#Note : might need to put syllablesplit as an attribute of the readability class in order to avoid duplicate code.
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
def GFI_score(text, statistics=None):
    """
    Outputs the Gunning fog index, a 1952 readability test estimating the years of formal education a person needs in order to understand a text on the first reading.
    The scale goes from 6 to 18, starting at the sixth grade in the United States.
        
    :param text: Content of a text, distincting between sentences.
    :type text: list(list(str)) or str 

    :return: The Gunning fog index of the current text
    :rtype: float
    """
    if statistics is not None:
        return 0.4*((statistics.totalWords/statistics.totalSentences) + 100*statistics.longWords/statistics.totalSentences)
    totalWords = 0
    totalSentences = len(text)
    longWords = 0
    for sent in text:
        totalWords += len(sent)
        longWords += len([token for token in sent if len(token)>6])
    if totalWords < 101:
        print("WARNING : Number of words is less than 100, This score is inaccurate")
    #NOTE : score is really innacurate regardless, since we divided by totalSentences instead of totalWords for the second ratio.
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


#TODO : When given self.statistics (or something equivalent since it's a corpus), bypass some calculations to optimize a bit further.
def traditional_scores(corpus):
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
    
    math_formulas_stddev = pd.DataFrame([stddev_GFI,stddev_ARI,stddev_FRE,stddev_FKGL,stddev_SMOG,stddev_REL],columns=levels)
    math_formulas_stddev.index = ["The Gunning fog index GFI", "The Automated readability index ARI","The Flesch reading ease FRE","The Flesch-Kincaid grade level FKGL","The Simple Measure of Gobbledygook SMOG","Reading Ease Level"]
    math_formulas_stddev.columns.name = "Std Dev"
    print(math_formulas_stddev)


    print("time elapsed perf counter:", time.perf_counter() - t0)
    return math_formulas
#this takes roughly half as much time.
