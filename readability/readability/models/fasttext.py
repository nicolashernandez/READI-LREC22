from . import models
from ..utils import utils
from ..parsed_collection import parsed_collection
import numpy as np
import ktrain
import os
from ktrain import text

def classify_corpus_fasttext(corpus, model_name = "fasttext"):
    """
    Imports, configures, and trains a fastText model.

    fastText is a library meant to be used for converting words into vector representations called word embeddings.
    In addition, the ktrain library is used as a wrapper over the deep learning library Tensorflow Keras to simplify the process.
    This allows to train a text classifier based on fastText word embeddings.

    :param corpus: Data input, preferably as a dict(class_label:list(text))
    :param str model_name: Choice of language model to use : fasttext, bigru, nbsvm
    :return: Classification task metrics, as detailed in ..models.compute_evaluation_metrics() for more details
    """
    if isinstance(corpus, parsed_collection.ParsedCollection):
        corpus_label_names = list(corpus.content.keys())
    else:
        corpus_label_names = list(corpus.keys())
    results_summary = list()

    # Uses the ktrain library to load the fastText model, then creates a learner object based on data split into train/test
    NUM_WORDS = 50000
    MAXLEN = 150
    NGRAMS_SIZE = 1
    x_train, y_train = utils.convert_corpus_to_list(corpus)

    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_array(x_train = x_train,
                        y_train = y_train,
                        class_names = corpus_label_names,
                        max_features = NUM_WORDS, 
                        maxlen = MAXLEN,
                        ngram_range = NGRAMS_SIZE,
                        val_pct = 0.1,
                        preprocess_mode = 'standard', # default
                        lang = "fr",
                        random_state = 2
                        )
    # Build and return a text classification model https://amaiya.github.io/ktrain/text/index.html#ktrain.text.text_classifier
    if model_name == "fasttext":
        model = text.text_classifier('fasttext', (x_train, y_train), preproc=preproc)
    elif model_name == "bigru":
        model = text.text_classifier('bigru', (x_train, y_train), preproc=preproc)
    elif model_name == "nbsvm":
        model = text.text_classifier('nbsvm', (x_train, y_train), preproc=preproc)

    # Returns a Learner instance that can be used to tune and train Keras models https://amaiya.github.io/ktrain/index.html#ktrain.get_learner
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))

    # Pseudo cross-validation by running n times the train/validation and resetting weights to its initial configuration
    # NOTE: True cross-validation can be done by recreating the learner instance with manual train/test split instead of just resetting weights.
    runs = 5
    results = list()

    init_weights = []
    for layer in learner.model.layers:
        init_weights.append(layer.get_weights()) # list of numpy arrays

    for RUN in range(runs):
        print ('-------------------------------------------------------run', RUN)
        # Properly reset the model's weights for a true cross-validation instead on fitting after its iterations
        for index in range(len(init_weights)):
            learner.model.layers[index].set_weights(init_weights[index])
        # train 
        learner.autofit(0.0001)
        # validate
        results.append(learner.validate(class_names=corpus_label_names))

    cm_init = [[0]*len(corpus_label_names)]*len(corpus_label_names)
    #return results
    for cm in results:
        cm_init += cm
    results_summary.append(cm_init)

    r = models.compute_evaluation_metrics(results_summary[0],round=2, data_name="", class_names=corpus_label_names)
    models.pp.pprint(r)

    return r

# -------------------- Ignore these, only used for reproducing READI paper contents -------------------

DATA_ENTRY_POINT = utils.DATA_ENTRY_POINT

def demo_getFastText(DATA_PATH, class_names):
    """Uses the ktrain library to load the fastText model, then creates a learner object with random test/train split based on csv file contents"""
    NUM_WORDS = 50000
    MAXLEN = 150
    #NGRAMS_SIZE = 1 # 8 minutes avec 2
    #NUM_WORDS = 80000
    #MAXLEN = 2000
    NGRAMS_SIZE = 1
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_csv(DATA_PATH,
                        'text',
                        label_columns = class_names,
                        val_filepath=None, # if None, 10% of data will be used for validation
                        max_features=NUM_WORDS, 
                        maxlen=MAXLEN,
                        ngram_range=NGRAMS_SIZE,
                        preprocess_mode='standard' # default
                        )

    model = text.text_classifier('fasttext', (x_train, y_train), preproc=preproc)
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))
    return (x_train, y_train), (x_test, y_test), preproc, model, learner

def demo_doFastText(name='ljl',test_flag = False):
    """Imports, configures, and trains the fastText model used in our paper.
    This method also prints the results in a latex-usable format, within the tabular tag.
    :param name: Which corpus data to use for reproducing results, can be "ljl","bibebook.com","JeLisLibre", or "all"
    :type name: str
    :return: Nothing, it just prints the execution trace and a latex-usable table
    :rtype: None
    """
    corpusnames = ['ljl']
    if name == 'all':
        corpusnames = ['ljl', 'bibebook.com', 'JeLisLibre']
    elif name == "bibebook.com":
        corpusnames = ['bibebook.com']
    elif name == "JeLisLibre":
        corpusnames = ['JeLisLibre']
    elif name != "ljl":
        raise ValueError("Please provide one of the following parameters instead : 'ljl', 'bibebook.com', 'JeLisLibre', or 'all'")

    results_summary = list()
    class_names_list = list()

    for CORPUSNAME in corpusnames:
        DATA_PATH = os.path.join(DATA_ENTRY_POINT,CORPUSNAME)+ '_hotvector.csv'
        class_names =  models.demo_get_csv_fieldnames(DATA_PATH)[2:]
        class_names_list.append(class_names)
        (x_train, y_train), (x_test, y_test), preproc, model, learner = demo_getFastText(DATA_PATH, class_names=class_names)
        # pseudo cross validation by running n times the train/validation
        number_of_run = 5
        results = list()

        if test_flag:
            init_weights = []
            for layer in learner.model.layers:
                init_weights.append(layer.get_weights()) # list of numpy arrays

        for RUN in range(number_of_run):
            print ('-------------------------------------------------------run', RUN)
            if test_flag:
                for index in range(len(init_weights)):
                    learner.model.layers[index].set_weights(init_weights[index])
            # train 
            #learner.autofit(0.00001)
            learner.autofit(0.0001)
            #learner.autofit(0.0007, 5)
            #learner.autofit(0.0001, 10)

            # validate
            print ('run', RUN, 'CORPUSNAME', CORPUSNAME, 'class_names', class_names)
            results.append(learner.validate(class_names=class_names))

        cm_init = [[0]*len(class_names)]*len(class_names)
        for cm in results:
            cm_init += cm
        results_summary.append(cm_init)
    print ('-------------------------------------------------------------')
    print ('total run', RUN)
    for i in range(len(corpusnames)):
        print ('CORPUSNAME', corpusnames[i])
        r = models.compute_evaluation_metrics(results_summary[i],round=2, data_name=corpusnames[i], class_names=class_names_list[i])
        models.pp.pprint(r)

        multicol_list = list()
        for j in range(len(class_names_list[i])):
            multicol_list.append('\multicolumn{3}{|c|}{'+ class_names_list[i][j]+'}')
        multicol =  '('+corpusnames[i]+') &'  + '&'.join(multicol_list) + '\\\\'
        header = '&' + 'P&\tR&\tF1&\t' * len(r['precision']) + 'Acc.&\tMacro avg.\\\\'
        line = list()
        for i in range (len(r['precision'])):
            line.append(str(r['precision'][i]))
            line.append(str(r['recall'][i]))
            line.append(str(r['fmeasure'][i]))
        line.append(str(r['accuracy']))
        line.append(str(r['macro_avg_fmeasure']))

        print(multicol)
        print(header)
        print('\t&'+'\t&'.join(line)+'\\\\')
        print()

    return 0

def demo_checkLR(name='ljl'):
    DATA_PATH = os.path.join(DATA_ENTRY_POINT,name)+ '_hotvector.csv'
  
    class_names =  models.demo_get_csv_fieldnames(DATA_PATH)[2:]
    print(class_names)    

    (x_train, y_train), (x_test, y_test), preproc, model, learner = demo_getFastText(DATA_PATH, class_names=class_names)

    learner.lr_find()
    learner.lr_plot()
