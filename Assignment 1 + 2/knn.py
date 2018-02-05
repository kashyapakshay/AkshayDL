from data_utils import load_CIFAR10
import numpy as np

class NearestNeighbor(object):
    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k=1):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in xrange(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argsort(distances)[:k]
            labels = []

            for index in min_index:
                labels.append(self.ytr[index])

            Ypred[i] = max(set(labels), key=labels.count)

        return Ypred

if __name__ == '__main__':
    Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py') # a magic function we provide

    Xtr, Ytr = Xtr[:100], Ytr[:100]
    Xte, Yte = Xte[:50], Yte[:50]

    # flatten out all images to be one-dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

    nn = NearestNeighbor() # create a Nearest Neighbor classifier class
    nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
    Yte_predict = nn.predict(Xte_rows, k=10) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
