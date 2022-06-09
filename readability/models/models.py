"""
The models module acts as an high-level interface for the use of language models.

It provides the following methods : 
Instantiate/load a model (most likely from Huggingface)
Customize/Tune the model
Train the model (and optionally save it)
Use the model for different tasks

It also includes several functions used to make the readi paper,
these are prefixed with demo_ and require a csv file in the hotvector format,
placed in the data directory
"""

import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

def demo_compute_evaluation_metrics(cm, round = -1, data_name = '', class_names=''): 
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

