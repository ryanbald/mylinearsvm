import mylinearsvm
import numpy as np
import sklearn.datasets, sklearn.svm
import sys

"""
python skl-comparision.py [--lambda lambda] [--C C] [--intercept]

    lambda: regularization parameter for my linear SVM
    C: regularization parameter for SKL's linear SVM
    intercept: flag to fit SKL's model with an intercept or not
"""

def _real_world_dataset():
    # make split reproducible
    np.random.seed(2)

    X, y = sklearn.datasets.load_digits(return_X_y=True)

    n = len(y)

    # remove features that have no variance over the full dataset
    extra_features = np.where(np.std(X, axis=0) == 0)
    X = np.delete(X, extra_features, axis=1)

    split = np.random.permutation(n)
    train_indices = split[:1350]
    test_indices = split[1350:]

    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]

    # standardize feature matrix and use same transformation on test data
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean)/X_std
    X_test = (X_test - X_mean)/X_std

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # hard code optimal regularization parameters for this dataset
    lambduh = 0.12
    C = 0.01

    # extract command line arguments
    fit_intercept = "--intercept" in sys.argv
    if "--lambda" in sys.argv:
        index = sys.argv.index("--lambda")
        lambduh = sys.argv[index + 1]
    if "--C" in sys.argv:
        index = sys.argv.index("--C")
        C = sys.argv[index + 1]

    # load digits dataset
    X_train, y_train, X_test, y_test = _real_world_dataset()

    linearSVM = mylinearsvm.MyLinearSVM(lambduh=float(lambduh))
    class_beta = linearSVM.fit_ovr(X=X_train, y=y_train)
    train_err = linearSVM.error_ovr(class_beta=class_beta, X=X_train, y=y_train)
    test_err = linearSVM.error_ovr(class_beta=class_beta, X=X_test, y=y_test)
    print("Lambda = {}".format(lambduh))
    print("Training Error: {}".format(train_err))
    print("Test Error: {}".format(test_err))

    print("")

    linearSVC = sklearn.svm.LinearSVC(C=float(C), fit_intercept=fit_intercept)
    linearSVC.fit(X_train, y_train)
    train_err = 1 - linearSVC.score(X_train, y_train)
    test_err = 1 - linearSVC.score(X_test, y_test)
    if fit_intercept:
        print("Fitting intercept")
    print("C = {}".format(C))
    print("Training Error (SKL): {}".format(train_err))
    print("Test Error (SKL): {}".format(test_err))
