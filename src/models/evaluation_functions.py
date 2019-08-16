import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from yellowbrick.classifier import ConfusionMatrix

def plot_confusion_matrix(model:sklearn.base.BaseEstimator,
                          X_train: np.ndarray,
                          X_test: np.ndarray,
                          y_train: np.ndarray,
                          y_test: np.ndarray):
    """
    Plots confusion matrix for given model and train/test data.
    Inputs:
        model: an sklearn classifier
        X_train: training examples
        X_test: test examples
        y_train: training labels corresponding to examples in X_train
        y_test: test labels corresponding to examples in X_test
    Returns: None
    """
    model_cm = ConfusionMatrix(model)
    model_cm.fit(X_train, y_train)
    model_cm.score(X_test, y_test)
    model_cm.poof()


def all_scores(y_true:np.ndarray,
               y_pred:np.ndarray):
    """
    Prints f-1 score, recall, and precision for the provided true values and predictions.
    Inputs:
        y_true: true labels of data
        y_pred: predicted labels of data
    Returns:
        test_recall: recall of y_true and y_pred
        test_precision: precision of y_true and y_pred
        test_f1: f1-score of y_true and y_pred
    """
    test_recall = recall_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred)
    test_f1 =  f1_score(y_true, y_pred)
    print(f'f-1 score: {f1_score(y_true, y_pred)}')
    print(f'recall: {recall_score(y_true, y_pred)}')
    print(f'precision:{precision_score(y_true, y_pred)}')
    return test_recall, test_precision, test_f1


def undersample(df:np.ndarray,
                target:np.ndarray,
                target_val:int=1,
                test_size:float=0.2):
    """
    Undersamples negative class such that the classes in the final dataset are balanced,
    and returns a train-test split of the data.
    Inputs:
        df: dataframe containing observations for training/validation
        target: Target variable in the dataframe
        target_val: desired value of the target variable to split as positive/negative class.
        test_size: proportion of the undersampled population that goes into the test set.
    Returns:
        X_train_downs: training set of examples of downsampled sample.
        X_test_downs: test set of examples of downsampled sample
        y_train_downs: training labels of examples in X_train_downs
        y_test_downs: training labels of examples in X_test_downs
    """
    pos_samples = df[df[target] == target_val]
    neg_samples = df[df[target] != target_val]

    num_pos = pos_samples.shape[0]
    num_neg = neg_samples.shape[0]

    downs_neg_idx = np.random.choice(
        neg_samples.index, size=num_pos, replace=False)
    downs_neg = neg_samples.loc[downs_neg_idx, :]
    downsample_df = pd.concat([downs_neg, pos_samples], axis=0)

    X_downs = downsample_df.drop('DEFAULT', axis=1)
    y_downs = downsample_df.DEFAULT

    X_train_downs, X_test_downs, y_train_downs, y_test_downs = train_test_split(
        X_downs, y_downs, test_size=test_size)
    return X_train_downs, X_test_downs, y_train_downs, y_test_downs


def upsample(df:np.ndarray,
             target:np.ndarray,
             target_val:int=1,
             test_size:float=0.2):
    """
    Upsamples positive class such that the classes in the final dataset are balanced,
    and returns a train-test split of the data.
    Inputs:
        df: dataframe containing observations for training/validation
        target: Target variable in the dataframe
        target_val: desired value of the target variable to split as positive/negative class.
        test_size: proportion of the oversampled population that goes into the test set.
    Returns:
        X_train_ups: training set of examples of upsampled sample.
        X_test_ups: test set of examples of upsampled sample
        y_train_ups: training labels of examples in X_train_ups
        y_test_ups: training labels of examples in X_test_ups
    """
    pos_samples = df[df[target] == target_val]
    neg_samples = df[df[target] != target_val]

    num_pos = pos_samples.shape[0]
    num_neg = neg_samples.shape[0]

    pos_samples_train, pos_samples_test = train_test_split(
        pos_samples, test_size=0.2)
    neg_samples_train, neg_samples_test = train_test_split(
        neg_samples, test_size=0.2)

    ups_pos_idx = np.random.choice(
        pos_samples_train.index, size=neg_samples_train.shape[0], replace=True)
    ups_pos = pos_samples_train.loc[ups_pos_idx, :]

    ccd_upsampled = pd.concat([ups_pos, neg_samples_train], axis=0)
    X_ups = ccd_upsampled.drop('DEFAULT', axis=1)
    y_ups = ccd_upsampled.DEFAULT

    ccd_ups_test = pd.concat([pos_samples_test, neg_samples_test], axis=0)
    X_ups_test = ccd_ups_test.drop('DEFAULT', axis=1)
    y_ups_test = ccd_ups_test.DEFAULT
    return X_ups, X_ups_test, y_ups, y_ups_test


def scores_of_best_search(trained_search: sklearn.model_selection._search.BaseSearchCV,
                          X_train: np.ndarray,
                          X_test: np.ndarray,
                          y_train: np.ndarray,
                          y_test: np.ndarray):
    """
    Outputs the precision, recall, and f-1 score of the best model in a trained search
    object(GridSearchCV, RandomizedCV) in sklearn.
    Inputs:
        trained_search: a fitted GridSearchCV or RandomizedCV object, with a
            best_estimator_ attribute.
        X_train: training set of examples
        X_test: test set of examples
        y_train: training set of labels, corresponding to X_train
        y_test: test set of labels, corresponding to X_test
    Returns:
        test_recall: recall of y_test and prediction of the best model
        test_precision: precision of y_test and prediction of the best model
        test_f1: f1 score of y_test and prediction of the best model
    """
    best_clf = trained_search.best_estimator_
    best_clf.fit(X_train, y_train)

    y_train_preds = best_clf.predict(X_train)
    print("Training Scores:")
    _, _2, _3 = all_scores(y_train, y_train_preds)

    best_preds = best_clf.predict(X_test)
    print("Test Scores:")
    test_recall, test_precision, test_f1 = all_scores(y_test, best_preds)
    return test_recall, test_precision, test_f1

def scores_confusion_matrix(trained_search: sklearn.model_selection._search.BaseSearchCV,
                            X_train: np.ndarray,
                            X_test: np.ndarray,
                            y_train: np.ndarray,
                            y_test: np.ndarray):
    """
    Outputs the score and confusion matrix of the best model in a trained search
    object(GridSearchCV, RandomizedCV) in sklearn.
    Inputs:
        trained_search: a fitted GridSearchCV or RandomizedCV object, with a
            best_estimator_ attribute.
        X_train: training set of examples
        X_test: test set of examples
        y_train: training set of labels, corresponding to X_train
        y_test: test set of labels, corresponding to X_test
    Returns:
        test_recall: recall of y_test and prediction of the best model
        test_precision: precision of y_test and prediction of the best model
        test_f1: f1 score of y_test and prediction of the best model
    """
    test_recall, test_precision, test_f1 = scores_of_best_search(trained_search, X_train, X_test, y_train, y_test)
    plot_confusion_matrix(trained_search.best_estimator_,
                          X_train, X_test, y_train, y_test)
    return test_recall, test_precision, test_f1
