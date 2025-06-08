"""
Module for generating predictions from machine learning models.

Giovanni Bishara - 869532
Singh Probjot - 869434
"""

import numpy as np

from keras.models import Sequential


def get_predictions(model, X_test):
    """
    Generate predictions for the provided test dataset using a specified model.


    Parameters:
    - model (Model): The model to be used for generating predictions. This can be any model that has a 
    predict method, like instances of keras.models.Sequential.
    - X_test (array-like): Test dataset on which predictions are to be made.
    
    Returns:
    - array-like: An array of predictions.
    to the nearest integer to represent class labels.
    """

    # Get the predictions
    y_pred = model.predict(X_test)

    if isinstance(model, Sequential):
        y_pred = np.round(y_pred)

    # Return the predictions
    return y_pred