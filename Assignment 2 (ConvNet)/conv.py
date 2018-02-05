import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def readimg(src='stinkbug.png'):
    return mpimg.imread(src)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def convolution(X, filt):
    # X = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,29,20], [21,22,23,24,25]])
    stride = 1
    padding = 0

    # X = np.random.rand(5, 5, 3)

    x_shape = X.shape
    f_shape = filt.shape

    feature_map = []

    for i in range(x_shape[0] - f_shape[0] + 1):
        row = []
        for j in range(x_shape[1] - f_shape[1] + 1):
            space = X[i:i+f_shape[0], j:j+f_shape[1]]
            row.append(np.sum(space * filt))
        feature_map.append(row)

    return X, feature_map

def ReLU(vect):
    return np.maximum(0, vect)

img = readimg(src='puppy.jpg')

filters = {
    'edge': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    'blur': np.array([[1,1,1], [1,1,1], [1,1,1]])
}

X, fm = convolution(img, filters['blur'])

# X2, fm2 = convolution(fm)
plt.imshow(ReLU(fm), cmap='gray')
plt.show()
