import numpy as np
import matplotlib.pyplot as plt
import util

np.seterr(all='raise')

factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, x, y):
        """Run solver to fit linear model. Update the value of
        self.theta using the normal equations.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        x.setflags(write=False) # make immutable
        self.theta = np.linalg.solve(x.T @ x, x.T @ y) # solve for theta using normal equations
        # *** END CODE HERE ***

    def create_poly(self, k, x):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        x_hat = np.zeros((x.shape[0], k + 1))

        for i in range(x.shape[0]):
            for j in range (k + 1):
                x_hat[i, j] = pow(x[i], j)

        return x_hat
        # *** END CODE HERE ***

    def create_sin(self, k, x):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        x_hat = np.zeros((x.shape[0], k + 2))

        for i in range(x.shape[0]):
            for j in range (k + 2):
                x_hat[i, j] = pow(x[i], j) if j < k + 1 else np.sin(x[i])

        return x_hat
        # *** END CODE HERE ***

    def predict(self, x):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        return x @ self.theta
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=None, filename='plot.png'):

    train_x, train_y = util.load_dataset(train_path, add_intercept=False)
    plot_x = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x, train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        model = LinearModel()
        x_hat = model.create_sin(k, train_x) if sine else model.create_poly(k, train_x)
        model.fit(x_hat, train_y)
        x_hat = model.create_sin(k, plot_x) if sine else model.create_poly(k, plot_x)
        plot_y = model.predict(x_hat)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x, plot_y, label='k=%d' % k)

    plt.title('Polynomial Regression')
    plt.legend()
    plt.show()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    run_exp(train_path, ks=[3])
    run_exp(train_path, ks=[3, 5, 10, 20])
    run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20])
    run_exp(small_path, ks=[1, 2, 5, 10, 20])
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
