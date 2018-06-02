import datetime
import numpy as np
import threading
import time


class MyLinearSVM:
    """Linear SVM implementing Huberized-hinge loss with one vs. rest multiclass classification"""

    def __init__(self, lambduh=1, max_iter=1000, epsilon=0.001, h=0.5, large=False, verbose=False):
        self.lambduh = lambduh
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.h = h
        self.verbose = verbose

        # if set, uses code that scales better
        self.large = large

    def _init_step_size(self):
        """Estimates initial step size for fitting (from slide 26 of week 3 slides)"""
        eigenvalues, _ = np.linalg.eig((self.X_train.T.dot(self.X_train))/self.n)
        estimate = 1/(max(eigenvalues) + self.lambduh)
        return estimate

    def _objective(self, beta, y, mask=None):
        """Calculates objective value of a given coefficient vector"""
        # if "mask" vector isn't already computed, compute it
        mask = 1 - y*self.X_train.dot(beta) if mask is None else mask

        if self.large:
            # more efficient for larger feature matrices
            empirical = 0
            for i in range(self.n):
                if mask[i] >= -self.h:
                    if mask[i] > +self.h:
                        empirical += mask[i]
                    else:
                        empirical += (mask[i] + self.h)**2/(4*self.h)
            empirical /= self.n

        else:
            # more efficient for smaller feature matrices
            empirical = np.zeros(self.n)
            piecewise_2 = np.where(abs(mask) <= self.h)
            piecewise_3 = np.where(mask > self.h)
            empirical[piecewise_2] = (mask[piecewise_2] + self.h)**2/(4*self.h)
            empirical[piecewise_3] = mask[piecewise_3]
            empirical = np.mean(empirical)

        regularization = self.lambduh*np.linalg.norm(beta)**2
        return empirical + regularization, mask

    def _gradient(self, beta, y, mask=None):
        """Calculates gradient of a given coefficient vector"""
        # if "mask" vector isn't already computed, compute it
        mask = 1 - y*self.X_train.dot(beta) if mask is None else mask

        if self.large:
            # more efficient for larger feature matrices
            empirical = 0
            for i in range(self.n):
                if mask[i] >= -self.h:
                    if mask[i] > +self.h:
                        empirical -= y[i]*self.X_train[i, :]
                    else:
                        empirical -= (y[i]*(mask[i] + self.h)*self.X_train[i, :])/(2*self.h)
            empirical /= self.n

        else:
            # more efficient for smaller feature matrices
            empirical = np.zeros((self.n, self.d))
            piecewise_2 = np.where(abs(mask) <= self.h)
            piecewise_3 = np.where(mask > self.h)

            # multiplies each row in the feature matrix by a scalar
            empirical[piecewise_2] = -((y[piecewise_2]*(mask[piecewise_2] + self.h))[:, np.newaxis]*self.X_train[piecewise_2])/(2*self.h)
            empirical[piecewise_3] = -(y[piecewise_3][:, np.newaxis]*self.X_train[piecewise_3])

            empirical = np.mean(empirical, axis=0)

        regularization = 2*self.lambduh*beta
        return empirical + regularization, mask

    def _backtracking(self, old_step_size, theta, theta_mask, grad_theta, norm_grad_theta, y, alpha=0.5, gamma=0.8):
        """Backtracking rule to select the next step size/coefficient vector (from slide 27 of week 3 slides)"""
        step_size = old_step_size
        obj_theta, _ = self._objective(theta, y, mask=theta_mask)
        potential_beta = theta - step_size*grad_theta
        obj_potential_beta, beta_mask = self._objective(potential_beta, y)
        while obj_potential_beta > obj_theta - alpha*step_size*norm_grad_theta**2:
            step_size *= gamma
            potential_beta = theta - step_size*grad_theta
            obj_potential_beta, beta_mask = self._objective(potential_beta, y)

        if self.verbose:
            print("objective value: {}".format(obj_potential_beta))

        return step_size, beta_mask

    def _alter(self, label):
        """Reduces the multi-label vector to two labels for one vs. rest multiclass classification"""
        altered = np.full(self.n, -1)
        altered[np.where(self.y_train == label)] = +1
        return altered

    def _fit_ovr_thread(self, ys, init_step_size, class_beta, indices):
        """Function passed to each thread for multithreading one vs. rest fitting"""
        for i, y in zip(indices, ys):
            beta = self._fit(init_step_size, y)

            # shared vector among threads to place the resulting coefficient vector
            class_beta[i, :] = beta

    def _fit(self, init_step_size, y):
        """Fits a single binary classifier, using the fast gradient algorithm with backtracking"""
        step_size = np.zeros(self.max_iter + 1)
        beta = np.zeros((self.max_iter + 1, self.d))
        theta = np.zeros((self.max_iter + 1, self.d))
        beta_mask = np.zeros((self.max_iter + 1, self.n))
        theta_mask = np.zeros((self.max_iter + 1, self.n))
        grad_beta = np.zeros((self.max_iter + 1, self.d))
        grad_theta = np.zeros((self.max_iter + 1, self.d))
        norm_grad_beta = np.zeros(self.max_iter + 1)
        norm_grad_theta = np.zeros(self.max_iter + 1)

        step_size[0] = init_step_size
        grad_beta[0, :], beta_mask[0, :] = self._gradient(beta[0, :], y)
        grad_theta[0, :], theta_mask[0, :] = self._gradient(theta[0, :], y)
        norm_grad_beta[0] = np.linalg.norm(grad_beta[0, :])
        norm_grad_theta[0] = np.linalg.norm(grad_theta[0, :])
        for t in range(self.max_iter):
            if self.verbose:
                print("ITERATION {}".format(t))

            step_size[t + 1], beta_mask[t + 1, :] = self._backtracking(step_size[t], theta[t], theta_mask[t, :], grad_theta[t, :], norm_grad_theta[t], y)
            beta[t + 1, :] = theta[t, :] - step_size[t + 1]*grad_theta[t, :]
            theta[t + 1, :] = beta[t + 1, :] + t/(t + 3)*(beta[t + 1, :] - beta[t, :])

            grad_beta[t + 1, :], _ = self._gradient(beta[t + 1, :], y, mask=beta_mask[t + 1, :])
            grad_theta[t + 1, :], theta_mask[t + 1, :] = self._gradient(theta[t + 1, :], y)
            norm_grad_beta[t + 1] = np.linalg.norm(grad_beta[t + 1, :])
            norm_grad_theta[t + 1] = np.linalg.norm(grad_theta[t + 1, :])

            if norm_grad_beta[t + 1] <= self.epsilon:
                break

        if self.verbose and t + 1 == self.max_iter:
            print("Maximum iterations reached")

        return beta[t + 1, :]

    def fit_ovr(self, X, y, threads=0, throttle=0):
        """Fits k binary classifiers"""

        """
            X: feature matrix to train with
            y: label vector to train with
            threads: number of threads to use (multithreading isn't used if set to 0)
            throttle: wait time between starting threads
        """
        self.X_train = X
        self.y_train = y

        self.n, self.d = self.X_train.shape
        self.labels = np.unique(self.y_train)
        self.k = len(self.labels)

        class_beta = np.zeros((self.k, self.d))
        init_step_size = self._init_step_size()

        if threads == 0:
            # No multithreading used
            for i, label in enumerate(self.labels):
                if self.verbose:
                    print("Training classifier {}...".format(i))

                altered_y = self._alter(label)
                beta = self._fit(init_step_size, altered_y)
                class_beta[i, :] = beta

                if self.verbose:
                    print("")

        else:
            classifiers_per_thread = int(np.ceil(self.k/threads))
            for t in range(threads):
                indices = range(t*classifiers_per_thread, min((t + 1)*classifiers_per_thread, self.k))
                altered_ys = [self._alter(self.labels[i]) for i in indices]
                target = self._fit_ovr_thread
                args = (altered_ys, init_step_size, class_beta, indices)
                thread = threading.Thread(target=target, args=args)
                thread.start()

                # delay before starting next thread
                if t < threads - 1:
                    time.sleep(throttle)

            # while the threads are still completing
            while threading.active_count() > 1:
                time.sleep(1)

        return class_beta

    def predict_ovr(self, class_beta, X):
        """Predicts labels using coefficient vectors for each classifier and a feature matrix"""

        """
            class_beta: coefficient vectors for all classifiers (k x d)
            X: feature matrix to use for prediction
        """
        n, _ = X.shape

        responses = np.zeros((n, self.k))
        for k in range(self.k):
            beta = class_beta[k, :]
            class_response = X.dot(beta)
            responses[:, k] = class_response

        # pick the labels that maximize the response (without an intercept)
        index_predictions = np.argmax(responses, axis=1)

        predictions = [self.labels[k] for k in index_predictions]
        return predictions

    def error_ovr(self, class_beta, X, y):
        """Calculates the misclassification error given the true label vector"""

        """
            class_beta: coefficient vectors for all classifiers (k x d)
            X: feature matrix to use for prediction
            y: true label vector
        """
        predictions = self.predict_ovr(class_beta, X)
        misses = [+1 if m != 0 else 0 for m in predictions - y]
        return np.mean(misses)
