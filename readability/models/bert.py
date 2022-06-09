from . import models
import random
import csv
import ktrain
from ktrain import text
import os

def demo_loadCorpusForTransformer(DATA_PATH, random_seed = 42):
  with open(DATA_PATH, 'r' ) as f:
    #reader = csv.DictReader(f)
    #x_train = [line['text'] for line in reader]
    csvreader = csv.reader(f)
    header = next(csvreader)
    print (header)
    lines = [line for line in csvreader]
    random.seed(random_seed)
    random.shuffle(lines)
    x = [line[1] for line in lines]
    y = [[int(x) for x in line[2:len(line)]] for line in lines]
  
    percent_train = 90
    len_train = round(len(lines)/100* percent_train)
    print ('len_train', len_train) 
    x_train = x[:len_train]
    x_test = x[len_train:]
    y_train = y[:len_train]
    y_test = y[len_train:]
    return x_train, x_test, y_train, y_test
    #for line in reader:
            # line is { 'workers': 'w0', 'constant': 7.334, 'age': -1.406, ... }
            # e.g. print( line[ 'workers' ] ) yields 'w0'
    #      print(line)


def demo_getTransformer(model_name, x_train, y_train, x_test, y_test, class_names, batch_size=6):
  t = text.Transformer(model_name, 
                     #maxlen=500, 
                     class_names=class_names #,
                     #label_columns = class_names,
                     #val_filepath=None, # if None, 10% of data will be used for validation
                      ##max_features=NUM_WORDS, 
                      #maxlen=MAXLEN,
                      #ngram_range=NGRAMS_SIZE,
                     # preprocess_mode='bert' 
                     )
#model = text.text_classifier('bert', (x_train, y_train) , preproc=preproc)
  trn = t.preprocess_train(x_train, y_train)
  val = t.preprocess_test(x_test, y_test)
  model = t.get_classifier() #     model (Model): A Keras Model instance
  learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=batch_size)
  print ('t',type(t), 
      '\ntrn', type(trn), 
      '\nval', type(val), 
      '\nmodel', type(model), 
      '\nlearner', type(learner))
  print (model.summary())
  return t, trn, val, model, learner


def demo_doBert(name='ljl'):
    #MODEL_NAME = 'distilbert-base-uncased'
    #MODEL_NAME = 'camembert-base'# https://huggingface.co/camembert-base ; https://camembert-model.fr/
    #MODEL_NAME = 'flaubert/flaubert_base_cased' # https://huggingface.co/flaubert/flaubert_base_cased
    # https://discuss.huggingface.co/t/helsinki-nlp-opus-mt-en-fr-missing-tf-model-h5-file/13467/3
    # 404 Client Error: Not Found for url: https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/tf_model.h5
    # bert https://github.com/amaiya/ktrain/blob/master/ktrain/text/models.py
    #MODEL_NAME = 'bert-base-multilingual-cased' # https://huggingface.co/bert-base-multilingual-cased ; https://github.com/deepset-ai/bert-tensorflow/blob/master/samples/bert_config.json
    #MODEL_NAME = 'bert-base-cased'

    corpusnames = ['ljl']
    if name == 'all':
        corpusnames = ['ljl', 'bibebook.com', 'JeLisLibre']
    elif name == "bibebook.com":
        corpusnames = ['bibebook.com']
    elif name == "JeLisLibre":
        corpusnames = ['JeLisLibre']
    elif name != "ljl":
        raise ValueError("Please provide one of the following parameters instead : 'ljl', 'bibebook.com', 'JeLisLibre', or 'all'")


    #model_names = ['bert-base-multilingual-cased', 'camembert-base', 'flaubert/flaubert_base_cased'] #]
    model_names = ['camembert-base' ] #]

    # PSEUDO CROSS VALIDATION by running several times the run and averaging the results
    # if number_of_run = -1 then lr_find, and fit_onecycle(2e-5, 1), fit_onecycle(5e-5, 1)
    number_of_run = 2

    for MODEL_NAME in model_names:
        print ('-------------------------------------------------------------------')

        results_summary = list()
        class_names_list = list()
        
        for CORPUSNAME in corpusnames:
            DATA_PATH = os.path.join(os.getcwd(),'data',CORPUSNAME)+ '_hotvector.csv'
            #DATA_PATH = os.getcwd()+'/gdrive/MyDrive/data/2022_readi/first-submission/'+ CORPUSNAME + '_hotvector.csv'
            #DATA_PATH = os.getcwd()+'/gdrive/MyDrive/data/'+ corpusname + '.csv'
            #DATA_PATH = corpusname + '_hotvector.csv'

            #class_names =  get_labels(DATA_PATH)   
            class_names =  models.demo_get_csv_fieldnames(DATA_PATH)[2:]
            class_names_list.append(class_names)

            x_train, x_test, y_train, y_test = demo_loadCorpusForTransformer(DATA_PATH)

            print ('CORPUS_NAME', CORPUSNAME, 'MODEL_NAME', MODEL_NAME, 'class_names', class_names)

            print ('--> getTransformer')
            t, trn, val, model, learner = demo_getTransformer(MODEL_NAME, x_train, y_train, x_test, y_test, class_names)

            # EXPLORATION
            if number_of_run <0:
                print ('--> lr_find')
                learner.lr_find(show_plot=True, max_epochs=2)

                #
                print ('--> fit_onecycle 2.5')
                learner.fit_onecycle(2e-5, 1) 
                learner.validate(class_names=t.get_classes())

                #
                print ('--> fit_onecycle 5e-5')
                learner.fit_onecycle(5e-5, 1) 
                learner.validate(class_names=t.get_classes())

            # PSEUDO CROSS VALIDATION by running n times the train/validation
            
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
                print ('MODEL_NAME',MODEL_NAME, 'run', RUN, 'CORPUSNAME', CORPUSNAME, 'class_names', class_names)
                results.append(learner.validate(class_names=class_names))

            cm_init = [[0]*len(class_names)]*len(class_names)
            for cm in results:
                cm_init += cm
            results_summary.append(cm_init)
            models.pp.pprint(models.demo_compute_evaluation_metrics(cm_init,round=2, data_name=CORPUSNAME, class_names=class_names))

            
        print ('-------------------------------------------------------------')
        print ('total run', RUN, 'MODEL_NAME',MODEL_NAME)
        for i in range(len(corpusnames)):
            print ('CORPUSNAME', corpusnames[i], 'CORPUSNAME', corpusnames[i])
            r = models.demo_compute_evaluation_metrics(results_summary[i],round=2, data_name=corpusnames[i], class_names=class_names_list[i])
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

            print (multicol)
            print (header)
            print ('\t&'+'\t&'.join(line)+'\\\\')
            print()
    return 0
