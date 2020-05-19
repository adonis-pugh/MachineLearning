import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set
    # Use np.savetxt to save predictions on eval set to save_path
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)

    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    y_eval = classifier.predict(x_eval)
    util.plot(x_eval, y_eval, classifier.theta, save_path)
    np.savetxt(save_path, y_eval)
    # *** END CODE HERE ***


def sigmoid(vec):
    """
    Parameters:

    `vec`: a numpy array or scalar
    ================================
    Returns the sigmoid function applied to each element of `vec`.
    """
    return 1 / (1 + np.exp(-vec))


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def hypothesis(self, x):
        return sigmoid(x @ self.theta)

    def gradient(self, x, y):
        return (y - self.hypothesis(x)) @ x

    def hessian(self, x):
        return -(x.T @ np.diag(self.hypothesis(x) * (1 - self.hypothesis(x)))) @ x

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        x.setflags(write=False)  # make immutable
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            last = self.theta[1]
            self.theta -= self.step_size * np.linalg.inv(self.hessian(x)) @ self.gradient(x, y)
            if self.verbose: print(self.theta) # print loss values
            if np.abs(self.theta[1] - last) < self.eps: break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        x.setflags(write=False) # make immutable
        predictions = [1 if self.hypothesis(x[i, :]) > 0.5 else 0 for i in range(x.shape[0])]
        return np.asarray(predictions)
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
