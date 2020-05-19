import numpy as np
import util
import sys
from random import random

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, valid_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_true = util.load_dataset(valid_path, add_intercept=True)
    x_transform, y_transform = upsample(x_train, y_train)

    logistic_regression(x_train, y_train, x_eval, y_true, output_path_naive) # vanilla
    logistic_regression(x_train, y_train, x_transform, y_transform, output_path_naive)
    logistic_regression(x_transform, y_transform, x_eval, y_true, output_path_upsampling) # re-sampled
    # *** END CODE HERE

def logistic_regression(x_train, y_train, x_eval, y_true, save_path):
    naive_clf = LogisticRegression()
    naive_clf.fit(x_train, y_train)
    y_eval = naive_clf.predict(x_eval)
    util.plot(x_eval, y_eval, naive_clf.theta, save_path)
    np.savetxt(save_path, y_eval)

    TP = np.sum(np.logical_and(y_eval == 1, y_true == 1)) # true positives
    TN = np.sum(np.logical_and(y_eval == 0, y_true == 0)) # true negatives
    FP = np.sum(np.logical_and(y_eval == 1, y_true == 0)) # false positives
    FN = np.sum(np.logical_and(y_eval == 0, y_true == 1)) # false negatives

    a = (TP + TN) / (TP + TN + FP + FN) # empirical accuracy
    a_0 = TN / (TN + FP) # negative accuracy
    a_1 = TP / (TP + FN) # positive accuracy
    a_bar = (a_0 + a_1) / 2 # balanced accuracy

    print("Accuracy: {}".format(a))
    print("Positive Accuracy: {}".format(a_1))
    print("Negative Accuracy: {}".format(a_0))
    print("Balanced Accuracy: {}".format(a_bar))

def upsample(x_train, y_train):
    x_positive = x_train[y_train == 1]
    x_negative = x_train[y_train == 0]
    y_positive = y_train[y_train == 1]
    y_negative = y_train[y_train == 0]
    p = y_positive.shape[0] / y_train.shape[0]
    kappa = p / (1 - p)
    x_positive = np.repeat(x_positive, int(1 / kappa), axis=0) # repeat positive examples
    y_positive = np.repeat(y_positive, int(1 / kappa)) # repeat positive labels
    x_transform = np.concatenate((x_positive, x_negative), axis=0)
    y_transform = np.concatenate((y_positive, y_negative))
    return x_transform, y_transform

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
