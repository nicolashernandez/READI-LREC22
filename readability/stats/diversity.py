"""The diversity module contains functions allowing to calculate notions related to text diversity. 
Currently it's only the text token ratio and noun-only version.
"""
import string


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
    # Convert if wrong type somehow + handle punctuation.
    if type(text) == list:
        doc = ' '.join(text)
    else:
        doc = text
    doc = doc.translate(str.maketrans('', '', string.punctuation))

    nb_unique = len(Counter(doc.split()))
    nb_tokens = len(doc.split())
    print("Returning Root TTR ratio = ",nb_unique,"/",nb_tokens**2,":",nb_unique/nb_tokens**2)
    return(nb_unique/nb_tokens**2)

# The following methods use the spacy "fr_core_news_sm" model to recognize lexical items.
def noun_token_ratio(text,nlp=None):
    """
    Outputs variant of the type token ratio, the TotalNoun/Noun Ratio.
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
    

    #TODO : check type of current nlp try catch
    print(type(nlp))

    nouns = [token.text for token in nlp(doc) if (not token.is_punct and token.pos_ == "NOUN")]
    nb_unique = len(Counter(nouns))
    nb_tokens = len(nouns)

    print("Returning Root TNNR ratio = ",nb_unique,"/",nb_tokens**2,":",nb_unique/nb_tokens**2)
    return(nb_unique/nb_tokens**2)
    