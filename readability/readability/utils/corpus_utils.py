#TODO : Repurpose these to be used on a list of texts (no notion of class)

# Utility functions that I don't think I can put in utils
def do_function_with_default_arguments(self,score_type):
    """Utility function that calls one of the function listed in self.score_types with default arguments"""
    if score_type in self.score_types["no_argument_needed"].keys():
        func = self.score_types["no_argument_needed"][score_type]
        print("WARNING: Score type '",score_type,"' not found in current instance, now using function ",func.__name__, " with default parameters",sep="")
        return func()
    elif score_type in self.score_types["argument_needed"].keys():
        func =  self.score_types["argument_needed"][score_type]
        print("WARNING: Score type '",score_type,"' not found in current instance, now using function ",func.__name__, " with default parameters",sep="")
        return func(score_type)

def get_corpus_scores(self, score_type=None, scores=None):
    """
    Utility function that searches relevant scores if a valid score_type that belongs to self.score_types is specified.
    If no scores can be recovered, this function will attempt to call the corresponding function with default arguments.

    :return: a dictionary, assigning a value to each text of each specified class of a corpus.
    :rtype: dict[str][str][str][str]
    """
    # There are 5 cases:
    # score_type is unknown : error
    # score_type is known,scores are known : do the actual function
    # score_type is known, scores are unknown, but stored inside corpus.stats : extract them then do actual function
    # score_type is known, scores are unkown, can't be found inside corpus_stats : get them w/ default then do actual function
    # score_type is known, scores are unknown, and not stored : get them w/ default param then do actual function

    if self.content_type == "text":
        raise TypeError('Content type is not corpus, please load something else to use this function.')
    if score_type not in list(self.score_types['no_argument_needed'].keys()) + list(self.score_types['argument_needed'].keys()):
        raise RuntimeError("the score type '", score_type ,"' is not recognized by the current Readability object, please pick one from", list(self.score_types['no_argument_needed'].keys()) + list(self.score_types['argument_needed'].keys()))
    if score_type is not None:
        if not hasattr(self,"corpus_statistics"):
            print("Suggestion : Use Readability.compile() beforehand to allow the Readability object to store information when using other methods.")
            if scores is not None:
                # Case : user provides scores even though .compile() was never used, trust these scores and perform next function.
                print("Now using user-provided scores from 'scores' argument")
                pass
            else:
                # Case : user provides no scores and cannot refer to self to find the scores, so use corresponding function with default parameters.
                scores = self.do_function_with_default_arguments(score_type)

        elif not hasattr(self.corpus_statistics[self.classes[0]][0],score_type):
            # Case : self.corpus_statistics exists, but scores aren't found when referring to self, so using corresponding function with default parameters.
            scores = self.do_function_with_default_arguments(score_type)
        else:
            # Case : scores found via referring to self.corpus_statistics, so just extract them.
            scores = {}
            for level in self.classes:
                scores[level] = []
                for score_dict in self.corpus_statistics[level]:
                    scores[level].append(score_dict.__dict__[score_type])

        return scores


def remove_outliers(self, stddevratio=1, score_type=None, scores=None):
    """
    Outputs a corpus, after removing texts which are considered to be "outliers", based on a standard deviation ratio
    A text is an outlier if its value value is lower or higher than this : mean +- standard_deviation * ratio
    In order to exploit this new corpus, you'll need to make a new Readability instance.
    For instance : new_r = Readability(r.remove_outliers(r.perplexity(),1))

    :return: a corpus, in a specific format where texts are represented as lists of sentences, which are lists of words.
    :rtype: dict[str][str][str][str]
    """
    scores = self.get_corpus_scores(score_type,scores)
    moy = list()
    for level in self.classes:
        inner_moy=0
        for score in scores[level]:
            inner_moy+= score/len(scores[level])
        moy.append(inner_moy)

    stddev = list()
    for index, level in enumerate(self.classes):
        inner_stddev=0
        for score in scores[level]:
            inner_stddev += ((score-moy[index])**2)/len(scores[level])
        inner_stddev = math.sqrt(inner_stddev)
        stddev.append(inner_stddev)

    outliers_indices = scores.copy()
    for index, level in enumerate(self.classes):
        outliers_indices[level] = [idx for idx in range(len(scores[level])) if scores[level][idx] > moy[index] + (stddevratio * stddev[index]) or scores[level][idx] < moy[index] - (stddevratio * stddev[index])]
        print("nb textes enleves(",level,") :", len(outliers_indices[level]),sep="")
        print(outliers_indices[level])

    corpus_no_outliers = copy.deepcopy(scores)
    for level in self.classes:
        offset = 0
        for index in outliers_indices[level][:]:
            corpus_no_outliers[level].pop(index - offset)
            offset += 1
        print("New number of texts for class", level, ":", len(corpus_no_outliers[level]))
    print("In order to use this new corpus, you'll have to make a new Readability instance.")
    return corpus_no_outliers


def stub_generalize_get_score(self, func, func_args , score_name):
    """
    Stub for functions that do the same logic when calculating a score and then potentially assign it to .corpus_statistics or else

    func is just a reference to the function
    func_args is a tuple  with the needed arguments for a text.
    score_name is the name of the score to assign to .corpus_statistics in order to re-use it in other methods.
    """
    if self.content_type == "text":
        print(func_args)
        print(*(func_args))
        return func(self.content, *(func_args))
    elif self.content_type == "corpus":
        scores = {}
        for level in self.classes:
            temp_score = 0
            scores[level] = []
            for index,text in enumerate(self.content[level]):
                scores[level].append(func(text, *(func_args)))
                temp_score += scores[level][index]
                if hasattr(self, "corpus_statistics"):
                    setattr(self.corpus_statistics[level][index],score_name,scores[level][index])
            temp_score = temp_score / len(scores[level])
            print("class", level, "mean score :" ,temp_score)
        return scores
    else:
        return -1

# NOTE: Maybe this should go in the stats subfolder to have less bloat.
def corpus_info(self):
    """
    Output several basic statistics such as number of texts, sentences, or tokens, alongside size of the vocabulary.
        
    :param corpus: Dictionary of lists of sentences (represented as a list of tokens)
    :type corpus: dict[class][text][sentence][token]
    :return: a pandas dataframe 
    :rtype: pandas.core.frame.DataFrame
    """
    if self.content_type != "corpus":
        raise TypeError("Current type is not recognized as corpus. Please provide a dictionary with the following format : dict[class][text][sentence][token]")
    else:
        # Extract the classes from the dictionary's keys.
        corpus = self.content
        levels = self.classes
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
                temp_nbr_ph = min(len(text),1) # NOTE : not sure this is a good idea.
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
