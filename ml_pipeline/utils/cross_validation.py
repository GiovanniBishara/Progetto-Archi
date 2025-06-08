"""
Module for performing cross-validation on three machine learning models.

This module includes functions to perform detailed cross-validation for these models:
Support Vector Machines (SVM) and Decision Trees.
Each function exploits Stratified 10-Fold Cross Validation.

Giovanni Bishara - 869532
Singh Probjot - 869434
"""

import numpy as np
import sklearn as sn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

from keras.models import Sequential

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

import os, ultraimport

get_predictions = ultraimport(f"{os.getcwd()}/../ml_pipeline/utils/predictions.py", "get_predictions")


# Metrics used by each fold during cross validation, needed for the SVM / Decision Tree function
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1_score': make_scorer(f1_score, average='macro')
}


def get_svm_dtc_scores(model, X, y):
    """
    Perform cross-validation on a model (either SVM or Decision Tree) using Stratified K-Fold with 10 splits.

    This function evaluates the specified model by using cross-validation to generate scores for accuracy, 
    precision, recall, and F1-score, which are then returned as separate lists for each metric.
    
    Parameters:
    - model (SVC or DecisionTreeClassifier): The model to be evaluated. Must be an instance of either 
    sklearn.svm.SVC or sklearn.tree.DecisionTreeClassifier.
    - X (array-like): Feature dataset for training and testing the model.
    - y (array-like): Target dataset corresponding to the features.
    
    Returns:
    - tuple of lists: Returns four lists containing the scores for accuracy, precision, recall, and F1-score 
    for each fold of the cross-validation process.
    """

    # This function only supports SVM and Decision Tree
    if not isinstance(model, svm.SVC) and not isinstance(model, DecisionTreeClassifier):
        return None
    
    # Stratified 10-fold cross validation is used
    cv=StratifiedKFold(n_splits=10, shuffle=True)

    # Perform the cross validation
    scores = sn.model_selection.cross_validate(model, X, y, cv=cv, scoring=scoring)

    # For each metric, retrieve the result of each fold
    accuracy_scores = scores['test_accuracy']
    precision_scores = scores['test_precision']
    recall_scores = scores['test_recall']
    f1_scores = scores['test_f1_score']

    # Return the scores
    return accuracy_scores, precision_scores, recall_scores, f1_scores