import numpy as np
import mylinearsvm
import sklearn.svm


def _simulated_dataset():
    # make data and split reproducible
    np.random.seed(46)

    n = 1600
    d = 160

    class1 = range(0, 200)
    class2 = range(200, 400)
    class3 = range(400, 600)
    class4 = range(600, 800)
    class5 = range(800, 1000)
    class6 = range(1000, 1200)
    class7 = range(1200, 1400)
    class8 = range(1400, 1600)

    X = np.random.rand(n, d)
    X[class1, :] = np.random.random() * X[class1, :] + np.random.random()
    X[class2, :] = np.random.random() * X[class2, :] + np.random.random()
    X[class3, :] = np.random.random() * X[class3, :] + np.random.random()
    X[class4, :] = np.random.random() * X[class4, :] + np.random.random()
    X[class5, :] = np.random.random() * X[class5, :] + np.random.random()
    X[class6, :] = np.random.random() * X[class6, :] + np.random.random()
    X[class7, :] = np.random.random() * X[class7, :] + np.random.random()
    X[class8, :] = np.random.random() * X[class8, :] + np.random.random()

    y = np.zeros(n)
    y[class1] = 1
    y[class2] = 2
    y[class3] = 3
    y[class4] = 4
    y[class5] = 5
    y[class6] = 6
    y[class7] = 7
    y[class8] = 8

    split = np.random.permutation(n)
    train_indices = split[:1200]
    test_indices = split[1200:]

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
    # create simulated dataset
    X_train, y_train, X_test, y_test = _simulated_dataset()

    # hard coded an optimal lambda and a larger epsilon to produce less output to the screen
    linearSVM = mylinearsvm.MyLinearSVM(lambduh=0.08, epsilon=0.01, verbose=True)

    class_beta = linearSVM.fit_ovr(X=X_train, y=y_train)
    train_err = linearSVM.error_ovr(class_beta=class_beta, X=X_train, y=y_train)
    test_err = linearSVM.error_ovr(class_beta=class_beta, X=X_test, y=y_test)

    print("")
    print("Training Error: {}".format(train_err))
    print("Test Error: {}".format(test_err))
