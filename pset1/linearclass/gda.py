import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)

    classifier = GDA()
    classifier.fit(x_train, y_train)
    y_eval = classifier.predict(x_eval)
    util.plot_gda(x_eval, y_eval, classifier.theta, classifier.theta_0, classifier.mu_0, classifier.mu_1, save_path)
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

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        self.theta_0 = 0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.mu_0 = None
        self.mu_1 = None

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        x.setflags(write=False)  # make immutable
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        phi = np.mean(y)
        mu_0 = np.sum(x[y == 0], axis=0) / y[y == 0].shape
        mu_1 = np.sum(x[y == 1], axis=0) / y[y == 1].shape
        mu_arr = np.full((x.shape[0], x.shape[1]), mu_0)
        mu_arr[y == 1] = mu_1
        sigma = ((x - mu_arr).T @ (x - mu_arr)) / x.shape[0]
        sigma_inv = np.linalg.inv(sigma)
        self.theta = -sigma_inv @ mu_0 + sigma_inv @ mu_1
        self.theta_0 = -np.log((1 - phi) / phi) + (mu_0.T @ sigma_inv @ mu_0) / 2 - (mu_1.T @ sigma_inv @ mu_1) / 2
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        x.setflags(write=False) # make immutable
        return (sigmoid((x @ self.theta) + self.theta_0) > 0.5).astype(int)
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
