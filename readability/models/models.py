"""
The models module acts as an high-level interface for the use of language models.

It should be able to provide the following :
Instantiate/load a model (most likely from Huggingface)
Customize/Tune the model
Train the model (and optionally save it)
Use the model for different tasks
"""


import numpy as np
import ktrain
import os
from ktrain import text
from csv import DictReader
import pprint
pp = pprint.PrettyPrinter(indent=4)

def compute_evaluation_metrics(cm, round = -1, data_name = '', class_names=''): 
  # ktrain learner.validate(class_names=class_names) return the confusion matrix
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
  # np.array([[2,1,0], [3,4,5], [6,7,8]])
  true_pos = np.diag(cm)
  false_pos = np.sum(cm, axis=0) - true_pos
  false_neg = np.sum(cm, axis=1) - true_pos
  precision = true_pos / (true_pos + false_pos)
  recall = true_pos / (true_pos + false_neg)
  support = true_pos + false_neg
  total_support = np.sum(support)
  #precision = true_pos / np.sum(cm, axis=0)
  #recall = true_pos / np.sum(cm, axis=1)
  #F-Measure = (2 * Precision * Recall) / (Precision + Recall)
  fmeasure = (2*precision*recall)/(precision+recall)
  accuracy = np.sum(true_pos)/np.sum(support)
  macro_avg_precision = np.average(precision)
  macro_avg_recall = np.average(recall)
  macro_avg_fmeasure = np.average(fmeasure)
  # weighted average (averaging the support-weighted mean per label),
  weighted_average_precision=np.sum(precision*support)/np.sum(support)
  weighted_average_recall=np.sum(recall*support)/np.sum(support)
  weighted_average_fmeasure=np.sum(fmeasure*support)/np.sum(support)
  results = dict()
  results['data_name'] = data_name
  results['class_names'] = class_names
  results['true_pos'] = true_pos
  results['false_pos'] = false_pos
  results['false_neg'] = false_neg
  results['support'] = support
  results['precision'] = precision
  results['recall'] = recall
  results['fmeasure'] = fmeasure
  results['accuracy'] = accuracy
  results['macro_avg_precision'] = macro_avg_precision
  results['macro_avg_recall'] = macro_avg_recall
  results['macro_avg_fmeasure'] = macro_avg_fmeasure
  results['total_support'] = total_support
  results['weighted_average_precision'] = weighted_average_recall
  results['weighted_average_recall'] = weighted_average_recall
  results['weighted_average_fmeasure'] = weighted_average_fmeasure
  
  if round>0:
    results['precision'] = np.round(precision, round)
    results['recall'] = np.round(recall, round)
    results['fmeasure'] = np.round(fmeasure, round)
    results['accuracy'] = np.round(accuracy, round)
    results['macro_avg_precision'] = np.round(macro_avg_precision, round)
    results['macro_avg_recall'] = np.round(macro_avg_recall, round)
    results['macro_avg_fmeasure'] = np.round(macro_avg_fmeasure, round)
    results['weighted_average_precision'] = np.round(weighted_average_precision, round)
    results['weighted_average_recall'] = np.round(weighted_average_recall, round)
    results['weighted_average_fmeasure'] = np.round(weighted_average_fmeasure, round)
        
  return results





def get_csv_fieldnames(DATA_PATH):
  # open file in read mode
  with open(DATA_PATH, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_dict_reader = DictReader(read_obj)
    # get column names from a csv file
    return csv_dict_reader.fieldnames




def getFastText (DATA_PATH, class_names="None"):
    NUM_WORDS = 50000
    MAXLEN = 150
    #NGRAMS_SIZE = 1# 1 # 8 minutes avec 2
    # nbsvm 
    #NUM_WORDS = 80000
    #MAXLEN = 2000
    NGRAMS_SIZE = 1
    # https://raw.githubusercontent.com/iampukar/toxic-comments-classification/master/train.csv
    # "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_csv(DATA_PATH,
                        'text',
                        label_columns = class_names,
                        val_filepath=None, # if None, 10% of data will be used for validation
                        max_features=NUM_WORDS, 
                        maxlen=MAXLEN,
                        ngram_range=NGRAMS_SIZE,
                        preprocess_mode='standard' # default
                        )
    # Build and return a text classification model https://amaiya.github.io/ktrain/text/index.html#ktrain.text.text_classifier
    model = text.text_classifier('fasttext', (x_train, y_train), preproc=preproc)
    #model = text.text_classifier('bigru', (x_train, y_train), preproc=preproc)
    #model = text.text_classifier('nbsvm', (x_train, y_train), preproc=preproc)

    # Returns a Learner instance that can be used to tune and train Keras models https://amaiya.github.io/ktrain/index.html#ktrain.get_learner
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))

    return (x_train, y_train), (x_test, y_test), preproc, model, learner



def doFastText():
    corpusnames = ['ljl'] #, 'bibebook.com', 'JeLisLibre']

    results_summary = list()
    class_names_list = list()

    for CORPUSNAME in corpusnames:
        DATA_PATH = os.getcwd()+'/gdrive/MyDrive/data/2022_readi/first-submission/'+ CORPUSNAME + '_hotvector.csv'
        #DATA_PATH = os.getcwd()+'/gdrive/MyDrive/data/'+ corpusname + '.csv'
        #DATA_PATH = corpusname + '_hotvector.csv'

        #class_names =  get_labels(DATA_PATH)   
        class_names =  get_csv_fieldnames   (DATA_PATH)[2:]
        class_names_list.append(class_names)
        #
        (x_train, y_train), (x_test, y_test), preproc, model, learner = getFastText(DATA_PATH, class_names=class_names)

        # pseudo cross validation by running n times the train/validation
        number_of_run = 5
        results = list()

        for RUN in range(number_of_run):
            print ('-------------------------------------------------------run', RUN)
            # train 
            # {0.05, 0.1, 0.25, 0.5}
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
        #print (cm_init)
        #pp.pprint(compute_evaluation_metrics(cm_init,round=2))
    print ('-------------------------------------------------------------')
    print ('total run', RUN)
    for i in range(len(corpusnames)):
        print ('CORPUSNAME', corpusnames[i])
        r = compute_evaluation_metrics(results_summary[i],round=2, data_name=corpusnames[i], class_names=class_names_list[i])
        pp.pprint(r)

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

        print (multicol)
        print (header)
        print ('\t&'+'\t&'.join(line)+'\\\\')


    return -1
def doBert():
    return -1