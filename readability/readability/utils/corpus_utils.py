#TODO : Repurpose these to be used on a list of texts (no notion of class)

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
