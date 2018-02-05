from data_utils import load_CIFAR10
import numpy as np

class SoftmaxClassifier:
    def __init__(self, delta=1.0, regularizer_const=1.0):
        self.delta = delta
        self.regularizer_const = regularizer_const

    def L_i(self, x_i, y_i, W):
        """
        Softmax Loss Function (Individual)

        Loss function for each individual data point.
        """

        W = np.array(W)
        x_i = np.array(x_i)
        y_i = np.array(y_i)

        scores = np.dot(W, x_i)
        scores -= np.max(scores) # normalize to avoid numerical instability

        actual_score = scores[y_i]
        # Softmax loss function
        loss = -actual_score + np.log(np.sum(np.exp(scores)))

        return loss

    def regularization_penalty(self, W):
        '''
        L2 Regularization
        '''

        reg_penalty = 0
        for m in xrange(W.shape[0]):
            for n in xrange(W.shape[1]):
                reg_penalty += (W[m][n])**2

        return reg_penalty

    def L(self, X, y, W):
        """
        Softmax Loss Function (Vectorized)

        Vectorized implemetation of loss function that can be applied to all training instances
        at once.
        """

        W = np.array(W)
        scores = np.dot(X, W.T)
        scores -= np.max(scores) # normalize to avoid numerical instability

        actual_score = scores[range(len(X)), y]

        # Softmax loss function
        loss = np.negative(actual_score) + np.log(np.sum(np.exp(scores)))

        average_data_loss = np.mean(loss)
        regularized_loss = average_data_loss + self.regularizer_const * self.regularization_penalty(W)

        return regularized_loss

    def randomSearch(self, X, y):
        best_loss = float('inf')
        best_W = np.zeros((10, 32 * 32 * 3))

        for i in xrange(250):
            W = np.random.randn(10, 32 * 32 * 3) * 0.0001
            loss = svm.L(X, y, W)

            if loss < best_loss:
                best_loss = loss
                best_W = W

        return best_W, best_loss

    def eval_numerical_gradient(self, f, x):
        """
        Computer numerical gradient of any function f at point(s) x
        """

        fx = f(x)
        gradient = np.zeros(x.shape)
        h = 0.00001

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            # Evaluate function at x+h
            ix = it.multi_index
            old_value = x[ix]
            x[ix] = old_value + h # Increment by h
            # We do this because f needs to be commputer over the entire range of points supplied
            fxh = f(x) # f(x + h)
            x[ix] = old_value # Restore to previous value

            # Compute the partial derivative to get slope
            gradient[ix] = (fxh - fx) / h
            it.iternext()

        return gradient

    def gradient_descent(self, X, y):
        print 'Gradient Descent...'

        # Since eval_numerical_gradient takes a single-argument function, we wrap the loss function
        # in another function that takes only Weights as arg and passes X and y for evaluating loss
        wrapped_loss_func = lambda W: self.L(X, y, W)

        W = np.random.rand(10, 3072) * 0.001 # Initialize weight vector

        print 'Weights Shape: ', W.shape, ' => ', np.product(W.shape), ' params to optimize'
        print 'Evaluating Gradient...'

        gradient = self.eval_numerical_gradient(wrapped_loss_func, W) # df - gradient

        loss_original = wrapped_loss_func(W) # the original loss
        print 'original loss: %f' % (loss_original, )

        # Try multiple step sizes. step_size is 10 raised to step_size_log.
        for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
            print 'Trying step size ', step_size_log

            step_size = 10 ** step_size_log
            W_new = W - (step_size * gradient)

            # Uncomment for verbose output about loss values
            loss_new = wrapped_loss_func(W_new)
            print 'for step size %f new loss: %f' % (step_size, loss_new)

            if loss_new < loss:
                W = W_new

        return W_new, loss_new

    def _train(self, X, y, optimizer_func):
        return optimizer_func(X, y)

    def train(self, X, y, optimizer='gradient', epochs=10):
        if optimizer is 'random':
            optimizer_func = self.randomSearch
        else:
            optimizer_func = self.gradient_descent

        return optimizer_func(X, y)


    def predict(self, x, W):
        return np.argmax(np.dot(W, x))

if __name__ == '__main__':
    print 'Loading CIFAR-10...\n'
    Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py') # a magic function we provide

    Xtr, Ytr = Xtr[:50], Ytr[:50]
    Xte, Yte = Xte[:20], Yte[:20]

    # flatten out all images to be one-dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

    softmax = SoftmaxClassifier() # create an SVN classifier class

    print 'Training...\n'
    W, loss = softmax.train(Xtr_rows, Ytr, 'gradient')
    correct = len(filter(lambda i: softmax.predict(Xte_rows[i], W) == Yte[i], xrange(len(Xte_rows))))

    # for i in xrange(len(Xte_rows)):
    #     print svm.predict(Xte_rows[i], W), ' - ', Yte[i]

    print 'accuracy: %.2f' % (float(correct) * 100/len(Yte))
