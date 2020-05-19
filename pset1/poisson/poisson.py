import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_true = util.load_dataset(eval_path, add_intercept=True)

    classifier = PoissonRegression(lr)
    classifier.fit(x_train, y_train)
    y_eval = classifier.predict(x_eval)
    util.plot_poisson(y_true, y_eval)
    np.savetxt(save_path, y_eval)
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        x.setflags(write=False) # make immutable
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            last = self.theta[1]
            self.theta += self.step_size * ((y @ x) - (np.exp(x @ self.theta) @ x))
            if self.verbose: print(self.theta) # print loss values
            if np.linalg.norm(self.theta[1] - last) < self.eps: break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        x.setflags(write=False) # make immutable
        return np.exp(x @ self.theta) # canonical response
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
