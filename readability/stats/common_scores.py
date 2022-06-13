"""
The common_scores module contains functions allowing to calculate GFI, ARI, FRE, FKGL, SMOG, and REL.
Can be improved by changing formulas/calculations depending on language.
"""
import math
from unidecode import unidecode
import pandas as pd
from scipy.stats import pearsonr
from .. import utils


# Text "must" be a list of sentences, which are lists of words.
def GFI_score(text, statistics=None):
    """
    Outputs the Gunning fog index, a 1952 readability test estimating the years of formal education needed to understand a text on the first reading.
    The scale goes from 6 to 18, starting at the sixth grade in the United States.
    The formula is : 0.4 * ( (words/sentences) + 100 * (complex words / words) )
        
    :param text: Content of a text, distincting between sentences.
    :type text: list(list(str)) or str 
    :param statistics: Refers to a readability.Statistics attribute, containing various pre-calculated information such as totalWords.
    :type statistics: readability.Statistics
    :return: The Gunning fog index of the current text
    :rtype: float
    """
    #FIXME : this score is wrong since we divided by totalSentences instead of totalWords for the second ratio. Leaving as-is for now.
    if statistics is not None:
        return 0.4*((statistics.totalWords/statistics.totalSentences) + 100*statistics.totalLongWords/statistics.totalSentences)
    totalWords = 0
    totalSentences = len(text)
    totalLongWords = 0
    for sent in text:
        totalWords += len(sent)
        totalLongWords += len([token for token in sent if len(token)>6])
    if totalWords < 101:
        print("WARNING : Number of words is less than 100, This score is inaccurate")
    
    score = 0.4*((totalWords/totalSentences) + 100*totalLongWords/totalSentences)
    return score

def ARI_score(text, statistics=None):
    """
    Outputs the Automated readability index, a 1967 readability test estimating the US grade level needed to comprehend a text
    The scale goes from 1 to 14, corresponding to age 5 to 18.
    The formula is 4.71 * (characters / words) + 0.5 (words / sentences) - 21.43
        
    :param text: Content of a text, distincting between sentences.
    :type text: list(list(str)) or str 
    :param statistics: Refers to a readability.Statistics attribute, containing various pre-calculated information such as totalWords.
    :type statistics: readability.Statistics
    :return: The Automated readability index of the current text
    :rtype: float
    """
    #FIXME : this score is wrong since we multiplied each ratio by 4.71 instead of doing it only for the first one.
    if statistics is not None:
        return 4.71*((statistics.totalCharacters/statistics.totalWords) + 0.5*statistics.totalWords/statistics.totalSentences)-21.43
    totalWords = 0
    totalSentences = len(text)
    totalCharacters = 0
    for sent in text:
        totalWords += len(sent)
        totalCharacters += sum(len(token) for token in sent)
    score = 4.71*((totalCharacters/totalWords) + 0.5*totalWords/totalSentences)-21.43
    return score

def FRE_score(text, statistics=None):
    """
    Outputs the Flesch reading ease, a 1975 readability test estimating the US school level needed to comprehend a text
    The scale goes from 100 to 0, corresponding to Grade 5 at score 100, up to post-college below score 30.
    The formula is 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
        
    :param text: Content of a text, distincting between sentences.
    :type text: list(list(str)) or str 
    :param statistics: Refers to a readability.Statistics attribute, containing various pre-calculated information such as totalWords.
    :type statistics: readability.Statistics
    :return: The Flesch reading ease of the current text
    :rtype: float
    """
    if statistics is not None:
        return 206.835-1.015*(statistics.totalWords/statistics.totalSentences)-84.6*(statistics.totalSyllables/statistics.totalWords)
    totalWords = 0
    totalSentences = len(text)
    totalSyllables = 0
    for sent in text:
        totalWords += len(sent)
        totalSyllables += sum(utils.syllablesplit(word) for word in sent)
    score_FRE = 206.835-1.015*(totalWords/totalSentences)-84.6*(totalSyllables/totalWords)
    return(score_FRE)

def FKGL_score(text, statistics=None):
    """
    Outputs the Flesch–Kincaid grade level, a 1975 readability test estimating the US grade level needed to comprehend a text
    The scale is meant to be a one to one representation, a score of 5 means that the text should be appropriate for fifth graders.
    The formula is 0.39 * (total words / total sentences)+11.8*(total syllables / total words) - 15.59
        
    :param text: Content of a text, distincting between sentences.
    :type text: list(list(str)) or str 
    :param statistics: Refers to a readability.Statistics attribute, containing various pre-calculated information such as totalWords.
    :type statistics: readability.Statistics
    :return: The Flesch–Kincaid grade level of the current text
    :rtype: float
    """
    if statistics is not None:
        return 0.39*(statistics.totalWords/statistics.totalSentences)+11.8*(statistics.totalSyllables/statistics.totalWords)-15.59
    totalWords = 0
    totalSentences = len(text)
    totalSyllables = 0
    for sent in text:
        totalWords += len(sent)
        totalSyllables += sum(utils.syllablesplit(word) for word in sent)
    score_FKGL = 0.39*(totalWords/totalSentences)+11.8*(totalSyllables/totalWords)-15.59
    return(score_FKGL)

def SMOG_score(text, statistics=None):
    """
    Outputs the Simple Measure of Gobbledygook, a 1969 readability test estimating the years of education needed to understand a text
    The scale is meant to be a one to one representation, a score of 5 means that the text should be appropriate for fifth graders.
    The formula is 1.043 * Square root (Number of polysyllables * (30 / number of sentences)) + 3.1291
        
    :param text: Content of a text, distincting between sentences.
    :type text: list(list(str)) or str 
    :param statistics: Refers to a readability.Statistics attribute, containing various pre-calculated information such as totalWords.
    :type statistics: readability.Statistics
    :return: The Simple Measure of Gobbledygook of the current text
    :rtype: float
    """
    # FIXME : the nbPolysyllables erroneously returns their own number of syllables instead of incrementing the counter by one.
    # Keeping as is for now
    if statistics is not None:
        return 1.043*math.sqrt(statistics.nbPolysyllables*(30/statistics.totalSentences))+3.1291
    totalSentences = len(text)
    nbPolysyllables = 0
    for sent in text:
        nbPolysyllables += sum(utils.syllablesplit(word) for word in sent if utils.syllablesplit(word)>=3)
        #nbPolysyllables += sum(1 for word in sent if utils.syllablesplit(word)>=3)
    score_SMOG = 1.043*math.sqrt(nbPolysyllables*(30/totalSentences))+3.1291
    return(score_SMOG)

def REL_score(text, statistics=None):
    """
    Outputs the Reading Ease Level, an adaptation of Flesch's reading ease for the French language,
    with changes to the coefficients taking into account the difference in length between French and English words.
    The formula is 207 - 1.015 * (Number of words / Number of sentences) - 73.6 * (Number of syllables / Number of words)
        
    :param text: Content of a text, distincting between sentences.
    :type text: list(list(str)) or str 
    :param statistics: Refers to a readability.Statistics attribute, containing various pre-calculated information such as totalWords.
    :type statistics: readability.Statistics
    :return: The Simple Measure of Gobbledygook of the current text
    :rtype: float
    """
    if statistics is not None:
        return 207-1.015*(statistics.totalWords/statistics.totalSentences)-73.6*(statistics.totalSyllables/statistics.totalWords)
    totalWords = 0
    totalSentences = len(text)
    totalSyllables = 0
    for sent in text:
        totalWords += len(sent)
        totalSyllables += sum(utils.syllablesplit(word) for word in sent)
    score_REL = 207-1.015*(totalWords/totalSentences)-73.6*(totalSyllables/totalWords)
    return(score_REL)

def traditional_scores(corpus, statistics=None):
    """
    Outputs a pandas dataframe containing the mean scores for various traditional readability measures.
    :param corpus: Dictionary of lists of sentences (represented as a list of tokens)
    :type corpus: dict[class][text][sentence][token]

    :return: a pandas dataframe 
    :rtype: pandas.core.frame.DataFrame
    """
    #Optimization : Calculate each score at once, since they share some parameters.
    levels = list(corpus.keys())
    GFI = {}
    ARI = {}
    FRE = {}
    FKGL = {}
    SMOG = {}
    REL = {}
    if statistics is not None:
        for level in levels:
            GFI[level] = []
            ARI[level] = []
            FRE[level] = []
            FKGL[level] = []
            SMOG[level] = []
            REL[level] = []
            for stats in statistics[level]:
                GFI[level].append(0.4*((stats.totalWords/stats.totalSentences) + 100*stats.totalLongWords/stats.totalSentences))
                ARI[level].append(4.71*((stats.totalCharacters/stats.totalWords) + 0.5*stats.totalWords/stats.totalSentences)-21.43)
                FRE[level].append(206.835-1.015*(stats.totalWords/stats.totalSentences)-84.6*(stats.totalSyllables/stats.totalWords))
                FKGL[level].append(0.39*(stats.totalWords/stats.totalSentences)+11.8*(stats.totalSyllables/stats.totalWords)-15.59)
                SMOG[level].append(1.043*math.sqrt(stats.nbPolysyllables*(30/stats.totalSentences))+3.1291)
                REL[level].append(207-1.015*(stats.totalWords/stats.totalSentences)-73.6*(stats.totalSyllables/stats.totalWords))
    else:
        for level in levels:
            GFI[level] = []
            ARI[level] = []
            FRE[level] = []
            FKGL[level] = []
            SMOG[level] = []
            REL[level] = []
            for text in corpus[level]:
                totalWords = 0
                totalLongWords = 0
                totalSentences = len(text)
                totalCharacters = 0
                totalSyllables = 0
                nbPolysyllables = 0
                for sent in text:
                    totalWords += len(sent)
                    totalLongWords += len([token for token in sent if len(token)>6])
                    totalCharacters += sum(len(token) for token in sent)
                    totalSyllables += sum(utils.syllablesplit(word) for word in sent)
                    nbPolysyllables += sum(utils.syllablesplit(word) for word in sent if utils.syllablesplit(word)>=3) 
                    #nbPolysyllables += sum(1 for word in sent if utils.syllablesplit(word)>=3)
                GFI[level].append(0.4*((totalWords/totalSentences) + 100*totalLongWords/totalSentences))
                ARI[level].append(4.71*((totalCharacters/totalWords) + 0.5*totalWords/totalSentences)-21.43)
                FRE[level].append(206.835-1.015*(totalWords/totalSentences)-84.6*(totalSyllables/totalWords))
                FKGL[level].append(0.39*(totalWords/totalSentences)+11.8*(totalSyllables/totalWords)-15.59)
                SMOG[level].append(1.043*math.sqrt(nbPolysyllables*(30/totalSentences))+3.1291)
                REL[level].append(207-1.015*(totalWords/totalSentences)-73.6*(totalSyllables/totalWords))
    
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

    # FIXME : FRE_list = [val/maxFRE for val in FRE_list]
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
    
    math_formulas_stddev = pd.DataFrame([stddev_GFI,stddev_ARI,stddev_FRE,stddev_FKGL,stddev_SMOG,stddev_REL],columns=levels)
    math_formulas_stddev.index = ["The Gunning fog index GFI", "The Automated readability index ARI","The Flesch reading ease FRE","The Flesch-Kincaid grade level FKGL","The Simple Measure of Gobbledygook SMOG","Reading Ease Level"]
    math_formulas_stddev.columns.name = "Standard Deviation values"
    print(math_formulas_stddev)

    return math_formulas
