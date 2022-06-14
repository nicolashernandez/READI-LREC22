"""
The methods module acts as an interface for Machine Learning applications (SVM/MLP and so on)

It should be able to provide the following :
Create the applications
Customize/Tune the applications
Use the applications
Export the results | Output visualisations
"""
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# dummy tokenizer for tf idf method
def demo_dummy_fun(doc):
    return doc

def convert_corpus_to_list(corpus):
    """Converts a dict[class][text][sentence][token] structure into list(str)"""
    corpus_as_list=list()
    labels_split = list()
    for level in corpus.keys():
        for text in corpus[level]:
          tex = []
          labels_split.append(list(corpus.keys()).index(level))
          for sent in text:
            for token in sent:
              tex.append(token.replace('\u200b',''))
          corpus_as_list.append(tex)
    return corpus_as_list, labels_split

def demo_doMethods(corpus, plot=False):
    # prep vectorzation
    tfidf_vectorizer = TfidfVectorizer(analyzer='word',
        tokenizer=demo_dummy_fun,
        preprocessor=demo_dummy_fun,
        token_pattern=None,
        min_df=2)

    # do vectorization
    temp_structure = convert_corpus_to_list(corpus)
    corpus_as_list = temp_structure[0]
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_as_list)
    labels_split = temp_structure[1]
    print ('Matrix dimensions:', tfidf_matrix.get_shape())

    # get all unique words in the corpus (the vocabulary and also the names of the matrix columns/features)
    vocabu = tfidf_vectorizer.get_feature_names()
    print ('Vocabulary size:', len(vocabu))

    # show document-term matrix
    tfidf_matrix = tfidf_matrix.toarray()
    #oops, get_feature_names_out is a feature of scikit above 1.0.0, except ktrain forces to download scikit 0.24.
    #let's see if this gives similar enough results..
    pd.DataFrame(tfidf_matrix, columns=tfidf_vectorizer.get_feature_names())

    # show detailed results for mlp
    print("MLP RESULTS")
    model = MLPClassifier(random_state=0)
    x = tfidf_matrix
    y = labels_split
    cvs=cross_val_score(model,x,y,scoring='accuracy',cv=5)
    print('cross-validation result for 5 runs =',cvs.mean())
    y_pred=cross_val_predict(model,x,y,cv=5)
    if plot:
        conf_mat=confusion_matrix(y_pred,y)
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=corpus.keys(), yticklabels=corpus.keys())
        plt.ylabel('Predicted')
        plt.xlabel('Actuals')
        plt.show()
    
    
    print(metrics.classification_report(y, y_pred, target_names=corpus.keys()))

    #show detailed results for svm
    print("SVM RESULTS")
    model = LinearSVC(random_state=0)
    x = tfidf_matrix
    y = labels_split
    cvs=cross_val_score(model,x,y,scoring='accuracy',cv=5)
    print('cross-validation result for 5 runs =',cvs.mean())
    y_pred=cross_val_predict(model,x,y,cv=5)
    if plot:
        conf_mat=confusion_matrix(y_pred,y)
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=corpus.keys(), yticklabels=corpus.keys())
        plt.ylabel('Predicted')
        plt.xlabel('Actuals')
        plt.show()
    
    print(metrics.classification_report(y, y_pred, target_names=corpus.keys()))







################Do these once reproducing paper is over################

#Configuration :


#Using a specific model :



#Using multiple models :

def stub_modelComparer():
    #prep/do models
    #models = [
    #RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    #LinearSVC(class_weight='balanced', random_state=0),
    #MultinomialNB(),
    #LogisticRegression(random_state=0), MLPClassifier(random_state=0)
    #]
    #CV = 5
    #cv_df = pd.DataFrame(index=range(CV * len(models)))
    #entries = []
    #for model in models:
    #    model_name = model.__class__.__name__
    #    accuracies = cross_val_score(model, tfidf_matrix, labels_split, scoring='accuracy', cv=CV)
    #    for fold_idx, accuracy in enumerate(accuracies):
    #        entries.append((model_name, fold_idx, accuracy))
    #cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    #if plot:
    #    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    #    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
    #                size=8, jitter=True, edgecolor="gray", linewidth=2)
    #    plt.show()

    ## show diff model general results output results
    #print(cv_df.groupby('model_name').accuracy.mean())
    return -1