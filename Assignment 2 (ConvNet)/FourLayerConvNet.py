import numpy as np

from data_utils import load_CIFAR10
from layers import *
from fast_layers import *

class FourLayerConvNet(object):
    """
    A four-layer convolutional network with the following architecture:

    [conv - relu - 2x2 max pool] - [conv - relu - 2x2 max pool] - [affine - relu] -[ affine - softmax]

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7, learning_rate=0.1,
               epochs=10, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.learning_rate = learning_rate
        self.epochs = epochs

        ############################################################################
        # Initialize weights and biases for the four-layer convolutional           #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.params['W1'] = np.random.normal(
            scale=weight_scale,
            size=(num_filters, input_dim[0], filter_size, filter_size)
        )

        W2_row_size = num_filters * input_dim[1]/2 * input_dim[2]/2
        self.params['W2'] = np.random.normal(
            scale=weight_scale,
            size=(num_filters, num_filters, filter_size, filter_size)
        )

        W3_row_size = num_filters * input_dim[1]/4 * input_dim[2]/4
        self.params['W3'] = np.random.normal(
            scale=weight_scale,
            size=(W3_row_size, hidden_dim)
        )

        self.params['W4'] = np.random.normal(
            scale=weight_scale,
            size=(hidden_dim, num_classes)
        )

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(num_filters)
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['b4'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def conv_relu_pool_forward(self, X, W, b, conv_param, pool_param):
        out_conv, cache_conv = conv_forward_fast(X, W, b, conv_param)
        out_relu, cache_relu = relu_forward(out_conv)
        out_pool, cache_pool = max_pool_forward_fast(out_relu, pool_param)

        return out_pool, (cache_conv, cache_relu, cache_pool)

    def affine_relu_forward(self, X, W, b):
        out_affine, cache_affine = affine_forward(X, W, b)
        out_relu, cache_relu = relu_forward(out_affine)

        return out_relu, (cache_affine, cache_relu)

    def conv_relu_pool_backward(self, dout, cache):
        cache_conv, cache_relu, cache_pool = cache

        ds = max_pool_backward_fast(dout, cache_pool)
        da = relu_backward(ds, cache_relu)
        dx, dw, db = conv_backward_fast(da, cache_conv)

        return dx, dw, db

    def affine_relu_backward(self, dout, cache):
        cache_affine, cache_relu = cache

        da = relu_backward(dout, cache_relu)
        dx, dw, db = affine_backward(da, cache_affine)

        return dx, dw, db

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the four-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # Implement the forward pass for the four-layer convolutional net,         #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        print 'Forward Pass...'
        out_1, cache_1 = self.conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_2, cache_2 = self.conv_relu_pool_forward(out_1, W2, b2, conv_param, pool_param)
        out_3, cache_3 = self.affine_relu_forward(out_2, W3, b3)
        out_4, cache_4 = affine_forward(out_3, W4, b4)
        scores = out_4

        loss, grads = 0, {}

        ############################################################################
        # Implement the backward pass for the four-layer convolutional net,        #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += sum(0.5 * self.reg * np.sum(W**2) for W in [W1, W2, W3, W4])

        print 'Backprop...'
        print '[1] Affine'
        dx_4, grads['W4'], grads['b4'] = affine_backward(dscores, cache_4)
        print '[2] Affine_ReLU'
        dx_3, grads['W3'], grads['b3'] = self.affine_relu_backward(dx_4, cache_3)
        print '[3] Conv_ReLU_Pool'
        dx_2, grads['W2'], grads['b2'] = self.conv_relu_pool_backward(dx_3, cache_2)
        print '[4] Conv_ReLU_Pool'
        dx_1, grads['W1'], grads['b1'] = self.conv_relu_pool_backward(dx_2, cache_1)

        grads['W4'] += self.reg * self.params['W4']
        grads['W3'] += self.reg * self.params['W3']
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']

        return loss, grads

    def train(self, X, y):
        print 'Training ConvNet:'
        print 'Training Examples: ', X.shape[0]

        for i in xrange(self.epochs):
            print '\nIteration %d:' % (i)

            loss, grads = self.loss(X, y)
            print 'Loss: ', loss

            self.params['W1'] += -self.learning_rate * grads['W1']
            self.params['W2'] += -self.learning_rate * grads['W2']
            self.params['W3'] += -self.learning_rate * grads['W3']
            self.params['W4'] += -self.learning_rate * grads['W4']

            self.params['b1'] += -self.learning_rate * grads['b1']
            self.params['b2'] += -self.learning_rate * grads['b2']
            self.params['b3'] += -self.learning_rate * grads['b3']
            self.params['b4'] += -self.learning_rate * grads['b4']

            print '\n'

        return loss

    def predict(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        out_1, cache_1 = self.conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_2, cache_2 = self.conv_relu_pool_forward(out_1, W2, b2, conv_param, pool_param)
        out_3, cache_3 = self.affine_relu_forward(out_2, W3, b3)
        out_4, cache_4 = affine_forward(out_3, W4, b4)
        scores = out_4

        return scores

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    # mask = range(num_training, num_training + num_validation)
    # X_val = X_train[mask]
    # y_val = y_train[mask]
    # mask = range(num_training)
    # X_train = X_train[mask]
    # y_train = y_train[mask]
    # mask = range(num_test)
    # X_test = X_test[mask]
    # y_test = y_test[mask]
    #
    # # Normalize the data: subtract the mean image
    # mean_image = np.mean(X_train, axis=0)
    # X_train -= mean_image
    # X_val -= mean_image
    # X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    # X_val = X_val.transpose(0, 3, 1, 2).copy()
    x_test = X_test.transpose(0, 3, 1, 2).copy()

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    Xtr, Ytr, Xte, Yte = get_CIFAR10_data() # a magic function we provide

    Xtr, Ytr = Xtr[:500], Ytr[:500]
    Xte, Yte = Xte[:200], Yte[:200]

    # flatten out all images to be one-dimensional
    # Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
    # Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

    convNet = FourLayerConvNet(epochs=10, learning_rate=0.01) # create an SVN classifier class

    loss = convNet.train(Xtr, Ytr)
    # correct = len(filter(lambda i: softmax.predict(Xte_rows[i], W) == Yte[i], xrange(len(Xte_rows))))

    print '\nPredicting...\n'

    predictions = convNet.predict(Xte)

    print np.argmax(predictions)
    print 'accuracy: %.2f' % (np.mean(np.argmax(predictions, axis=1) == Yte))
