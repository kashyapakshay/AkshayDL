import numpy as np

def ReLU(vect):
    return np.maximum(0, vect)

def sigmoid(vect):
    return 1.0 / (1.0 + np.exp(-vect))

def softmax(vect):
    expVect = np.exp(vect)
    # Compute softmax across rows (axis = 1); compatible with matrices.
    return expVect / np.sum(expVect, axis=1, keepdims=True)

# GENERATE NON-LINEAR, SPIRAL-LOOKING DATA
# Borrowed from the Stanford Course Notes page
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

# 1. INITIALIZE PARAMETERS RANDOMLY
# Hidden layer weights and biases
W1 = np.random.randn(D, N) * 0.01
b1 = np.zeros((1, N))
W2 = np.random.randn(N, N) * 0.01
b2 = np.zeros((1, N))
W3 = np.random.randn(N, K) * 0.01
b3 = np.zeros((1, K))

# Hyperparameters
stepSize = 1
regularizationStrength = 0.001

# Activation function
# Can also use Sigmoid, Tanh, Leaky ReLU
activation = ReLU

# 2. GRADIENT DESCENT LOOP
trainingSize = X.shape[0]
epochs = 10000

print '2-FULLY CONNECTED LAYERS\n(ACTIVATION: ReLU)\n'

for i in range(epochs):
    # 3. FEED-FORWARD COMPUTATION
    h1 = activation(np.dot(X, W1) + b1)
    h2 = activation(np.dot(h1, W2) + b2)
    out = np.dot(h2, W3) + b3

    # 4. COMPUTE CLASS SCORES, PROBABILITIES
    scores = out
    probabilities = softmax(scores)

    # 5. CALCULATE CROSS-ENTROPY LOSS + L2 REGULARIZATION
    correctProbabilities = -np.log(probabilities[range(trainingSize), y])
    dataLoss = np.sum(correctProbabilities) / trainingSize
    regularization = 0.5 * regularizationStrength * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    totalLoss = dataLoss + regularization

    if i % 1000 == 0:
        print '[Epoch %5d] Total Loss: %f' % (i, totalLoss)

    # 6. COMPUTE GRADIENT ON SCORES, THE FINAL OUTPUT.
    # THIS IS WHERE WE START BACKPROPAGATION FROM, THE FINAL OUTPUT OF THE NETWORK.
    dscores = probabilities
    dscores[range(trainingSize), y] -= 1
    dscores /= trainingSize

    # 7. BACKPROPAGATE GRADIENT TO WEIGHTS, BIASES, OTHER PARAMETERS, IF ANY.
    dW3 = np.dot(h2.T, dscores)
    db3 = np.sum(dscores, axis=0, keepdims=True)
    dhidden2 = np.dot(dscores, W3.T)
    dhidden2[h2 <= 0] = 0

    dW2 = np.dot(h1.T, dhidden2)
    db2 = np.sum(dhidden2, axis=0, keepdims=True)
    dhidden1 = np.dot(dhidden2, W2.T)
    dhidden1[h1 <= 0] = 0

    dW1 = np.dot(X.T, dhidden1)
    db1 = np.sum(dhidden1, axis=0, keepdims=True)

    # 8. ADD REGULARIZATION TO GRADIENT AS WELL. LAMBDA * REGULARIZATION_STRENGTH
    dW3 += regularizationStrength * W3
    dW2 += regularizationStrength * W2
    dW1 += regularizationStrength * W1

    # 9. UPDATE PARAMETERS DEPENDING ON GRADIENTS BACKPROP GAVE US.
    W1 += -stepSize * dW1
    b1 += -stepSize * db1
    W2 += -stepSize * dW2
    b2 += -stepSize * db2
    W3 += -stepSize * dW3
    b3 += -stepSize * db3

# EVALUTE
h1 = activation(np.dot(X, W1) + b1)
h2 = activation(np.dot(h1, W2) + b2)
scores = np.dot(h2, W3) + b3
predictions = np.argmax(scores, axis=1)
print '\nAccuracy: %.2f' % (np.mean(predictions == y))
