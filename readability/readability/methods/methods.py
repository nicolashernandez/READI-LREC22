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

from ..utils import utils
from ..parsed_collection import parsed_collection


#Configuration :

#Using a specific model :
def classify_corpus_MLP(corpus, plot=False):
    """
    Uses a MLP (Multilayer perceptron) model to classify a given collection of texts.

    While not directly related to readability, this can be used to exploit the features acquired from research into readability, to try and see
    if these can correlate with readability one way or another.

    :param bool plot: Whether to graphically output the results to the current user's terminal or application.
    :return: a string showing relevant metrics after performing a text classification task.
    :rtype: sklearn.metrics.classification_report
    """
        
    if isinstance(corpus, parsed_collection.ParsedCollection):
        corpus_label_names = corpus.content.keys()
    else:
        corpus_label_names = corpus.keys()
    
    # prep vectorzation
    tfidf_vectorizer = prepare_tfidf_vectorizer(tokenizer=None)

    # do vectorization
    tfidf_matrix, labels = prepare_tf_idf_matrix(corpus,tfidf_vectorizer)

    # Use MLP
    print("Now using Multilayer Perceptron to attempt to classify corpus")
    model = MLPClassifier(random_state=0)
    x = tfidf_matrix ; y = labels
    cvs=cross_val_score(model,x,y,scoring='accuracy',cv=5)
    print('cross-validation result for 5 runs =',cvs.mean())
    y_pred=cross_val_predict(model,x,y,cv=5)
    if plot:
        conf_mat=confusion_matrix(y_pred,y)
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=corpus_label_names, yticklabels=corpus_label_names)
        plt.ylabel('Predicted')
        plt.xlabel('Actuals')
        plt.show()
    
    print(metrics.classification_report(y, y_pred, target_names=corpus_label_names))
    return metrics.classification_report(y, y_pred, target_names=corpus_label_names)

def classify_corpus_SVM(corpus, plot=False):
    """
    Uses a SVM (Support Vector Machine) model to classify the given collection of texts.

    While not directly related to readability, this can be used to exploit the features acquired from research into readability, to try and see
    if these can correlate with readability one way or another.

    :param bool plot: Whether to graphically output the results to the current user's terminal or application.
    :return: a string showing relevant metrics after performing a text classification task.
    :rtype: sklearn.metrics.classification_report
    """
    if isinstance(corpus, parsed_collection.ParsedCollection):
        corpus_label_names = corpus.content.keys()
    else:
        corpus_label_names = corpus.keys()
    
    # prep vectorzation
    tfidf_vectorizer = prepare_tfidf_vectorizer(tokenizer=None)

    # do vectorization
    tfidf_matrix, labels = prepare_tf_idf_matrix(corpus,tfidf_vectorizer)

    # Use SVM
    print("Now using Support Vector Machine to attempt to classify corpus")
    model = LinearSVC(random_state=0)
    x = tfidf_matrix ; y = labels
    cvs=cross_val_score(model,x,y,scoring='accuracy',cv=5)
    print('cross-validation result for 5 runs =',cvs.mean())
    y_pred=cross_val_predict(model,x,y,cv=5)
    if plot:
        conf_mat=confusion_matrix(y_pred,y)
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=corpus_label_names, yticklabels=corpus_label_names)
        plt.ylabel('Predicted')
        plt.xlabel('Actuals')
        plt.show()
    
    print(metrics.classification_report(y, y_pred, target_names=corpus_label_names))
    return metrics.classification_report(y, y_pred, target_names=corpus_label_names)


#Using multiple models:
def compare_models(corpus, plot=True):
    """
    Uses several popular Machine Learning models to classify the given collection of texts, to show which ones currently performs the best.

    Uses a Random Forest Classifier, a SVM (Support Vector Machine) model, a Multinomial Naive Bayes model, Logistic Regression, and a Multilayer
    perceptron to see which ones currently performs the text classifiction task the best.
    """
    # prep vectorzation
    tfidf_vectorizer = prepare_tfidf_vectorizer(tokenizer=None)

    # do vectorization
    tfidf_matrix, labels = prepare_tf_idf_matrix(corpus,tfidf_vectorizer)

    #prep models
    models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(class_weight='balanced', random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0), MLPClassifier(random_state=0)
    ]

    # do models
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, tfidf_matrix, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    if plot:
        sns.boxplot(x='model_name', y='accuracy', data=cv_df)
        sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                    size=8, jitter=True, edgecolor="gray", linewidth=2)
        plt.show()

    # show diff model general results output results
    print(cv_df.groupby('model_name').accuracy.mean())
    return -1

# -------------------- Ignore these, only used for reproducing READI paper contents -------------------

def dummy_fun(doc):
    """dummy tokenizer for tf-idf method, since we already tokenize our texts thanks to the ReadabilityProcessor"""
    return doc

def prepare_tfidf_vectorizer(tokenizer=None):
    """Returns a tfidf vectorizer in order to prepare a tfidf matrix representing the contents of each class of a text collection"""
    if tokenizer is None:
        tfidf_vectorizer = TfidfVectorizer(analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        min_df=2)
    else:
        tfidf_vectorizer = TfidfVectorizer(analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        min_df=2)
    return tfidf_vectorizer

def prepare_tf_idf_matrix(corpus, tfidf_vectorizer):
    """Returns a tfidf matrix, representing the contents of each class of a text collection."""
    temp_structure = utils.convert_corpus_to_list(corpus)
    corpus_as_list = temp_structure[0]
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_as_list)
    #print ('Matrix dimensions:', tfidf_matrix.get_shape())
    tfidf_matrix = tfidf_matrix.toarray()
    labels = temp_structure[1]
    return tfidf_matrix, labels

def demo_doMethods(corpus, plot=False):
    """Prepare a tfidf matrix, then calls an MLP classifier, then an SVM classifier."""
    if isinstance(corpus, parsed_collection.ParsedCollection):
        corpus_label_names = corpus.content.keys()
    else:
        corpus_label_names = corpus.keys()
    
    # prep vectorzation
    tfidf_vectorizer = prepare_tfidf_vectorizer(tokenizer=None)

    # do vectorization
    tfidf_matrix, labels = prepare_tf_idf_matrix(corpus,tfidf_vectorizer)

    # show detailed results for mlp
    print("MLP RESULTS")
    model = MLPClassifier(random_state=0)
    x = tfidf_matrix
    y = labels
    cvs=cross_val_score(model,x,y,scoring='accuracy',cv=5)
    print('cross-validation result for 5 runs =',cvs.mean())
    y_pred=cross_val_predict(model,x,y,cv=5)
    if plot:
        conf_mat=confusion_matrix(y_pred,y)
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=corpus_label_names, yticklabels=corpus_label_names)
        plt.ylabel('Predicted')
        plt.xlabel('Actuals')
        plt.show()
    print(metrics.classification_report(y, y_pred, target_names=corpus_label_names))

    #show detailed results for svm
    print("SVM RESULTS")
    model = LinearSVC(random_state=0)
    x = tfidf_matrix
    y = labels
    cvs=cross_val_score(model,x,y,scoring='accuracy',cv=5)
    print('cross-validation result for 5 runs =',cvs.mean())
    y_pred=cross_val_predict(model,x,y,cv=5)
    if plot:
        conf_mat=confusion_matrix(y_pred,y)
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=corpus_label_names, yticklabels=corpus_label_names)
        plt.ylabel('Predicted')
        plt.xlabel('Actuals')
        plt.show()
    print(metrics.classification_report(y, y_pred, target_names=corpus_label_names))
