# Image Captioning using Convolutional and Recurrent Neural Networks.
# Author: Akshay Kashyap, CS '18, Union College, NY.
#
# Based on talk and paper by Andrej Karpathy and Fei-Fei Li at Stanford.
# Talk: https://cs.stanford.edu/people/karpathy/sfmltalk.pdf
# Paper: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf
# Architecture as suggested in the slides.
# Uses the 16-layer VGG architecture by K. Simonyan, A. Zisserman
# at Oxford: http://arxiv.org/pdf/1409.1556
# Uses generic GRU RNN, as given by Keras example:
# https://keras.io/getting-started/sequential-model-guide/#examples
#
# Usage: python ImageCaptioner.py

from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.core import Dropout, Activation, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Embedding, GRU, TimeDistributed, RepeatVector, Merge
from keras.optimizers import SGD
from keras import backend as K

import numpy as np
import cv2

class ConvNetFactory(object):
    '''Factory class to build a ConvNet model.'''

    def __init__(self):
        self.architectureBuilders = {
            'VGG16': self._buildVGG16
        }

    def buildModel(self, architecture='VGG16', preTrainedWeights=None):
        if architecture in self.architectureBuilders:
            return self.architectureBuilders[architecture](preTrainedWeights=preTrainedWeights)

    def _buildVGG16(self, preTrainedWeights=None):
        ''' Build VGG-16 (OxfordNet) Architecture Model.'''

        model = Sequential()

        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        if preTrainedWeights:
            model.load_weights(preTrainedWeights)

        return model

class RecurrentNetFactory:
    '''Factory class to build a RecurrentNet model.'''

    def __init__(self):
        self.architectureBuilders = {
            'Generic': self._buildGeneric
        }

    def buildModel(self, architecture='Generic', preTrainedWeights=None, inputDimension=50, maxInputLength=50):
        if architecture in self.architectureBuilders:
            return self.architectureBuilders[architecture](inputDimension, maxInputLength, preTrainedWeights)

    def _buildGeneric(self, inputDimension, maxInputLength, preTrainedWeights):
        '''Generic GRU-based RNN.'''
        model = Sequential()

        model.add(Embedding(inputDimension, 256, input_length=maxInputLength))
        model.add(GRU(output_dim=128, return_sequences=True))
        model.add(TimeDistributed(Dense(128)))

        return model

class ImageCaptionModel:
    def __init__(self):
        self.convNetFactory = ConvNetFactory()
        self.recurrentNetFactory = RecurrentNetFactory()

    def buildImageCaptioner(self, vocabSize=50, maxCaptionLength=50, convNetArchitecture='VGG16', recurrentNetArchitecture='Generic'):
        CNN = self.convNetFactory.buildModel(architecture=convNetArchitecture)
        RNN = self.recurrentNetFactory.buildModel(architecture=recurrentNetArchitecture, inputDimension=vocabSize, maxInputLength=maxCaptionLength)

        # Repeat CNN output as many times as the max caption length.
        CNN.add(RepeatVector(maxCaptionLength))

        if convNetArchitecture == 'VGG16':
            # Remove last two FC layers since we will append CNN to RNN
            CNN.layers.pop()
            CNN.layers.pop()

        # Merge the CNN and RNN
        imageCaptionModel = Sequential()
        imageCaptionModel.add(Merge([CNN, RNN], mode='concat', concat_axis=-1))
        # Append a GRU cell at the end.
        imageCaptionModel.add(GRU(256, return_sequences=False))
        # Fully-Connected layer at the end with Softmax activation.
        imageCaptionModel.add(Dense(vocabSize))
        imageCaptionModel.add(Activation('softmax'))

        return imageCaptionModel

class ImageCaptioner:
    def __init__(self, images, captions, dims=(3, 224, 224), convNetArchitecture='VGG16', recurrentNetArchitecture='Generic'):
        K.set_image_dim_ordering('th')

        self.images = images
        self.captions = captions

        words = [caption.split() for caption in self.captions]
        self.vocabSize, self.word2Index, self.index2Word = self.buildVocab(words)

        counts = [len(caption) for caption in self.captions]
        self.maxCaptionLength = max(counts)

        self.imageCaptionModel = ImageCaptionModel().buildImageCaptioner(
            vocabSize=self.vocabSize,
            maxCaptionLength=self.maxCaptionLength,
            convNetArchitecture=convNetArchitecture,
            recurrentNetArchitecture=recurrentNetArchitecture
        )

    def buildVocab(self, words):
        '''Builds a vocabulary of words, which is a list all of unique words that appear in the
        caption. Returns a mapping of given words to their indices.'''

        uniqueWords = []
        for word in words:
            uniqueWords.extend(word)
        uniqueWords = list(set(uniqueWords))

        word2Index = {}
        index2Word = {}

        for index, word in enumerate(uniqueWords):
            word2Index[word] = index
            index2Word[index] = word

        vocabSize = len(word2Index)

        return vocabSize, word2Index, index2Word

    def train(self):
        # Build portions of captions that we will use to train the ImageCaptioner on.
        # Given a portion, the model predicts the next word in the sequence.
        captionPortions = []
        for text in self.captions:
            one = [self.word2Index[txt] for txt in text.split()]
            captionPortions.append(one)

        captionPortions = sequence.pad_sequences(captionPortions, maxlen=self.maxCaptionLength, padding='post')
        nextWords = np.zeros((len(self.captions), self.vocabSize))

        for index, caption in enumerate(self.captions):
            trueIndices = np.asarray([self.word2Index[word] for word in caption.split()])
            nextWords[index, trueIndices] = 1

        self.imageCaptionModel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.imageCaptionModel.fit([self.images, captionPortions], nextWords, batch_size=1, nb_epoch=5)
        self.imageCaptionModel.save('imcap-class.h5')

    def predict(self, image, initialCaption):
        indexSequence = [self.word2Index[word] for word in initialCaption.split()]
        indexSequence = indexSequence + [0]*(self.maxCaptionLength - len(indexSequence))

        prediction = self.imageCaptionModel.predict([np.array([image]), np.array([indexSequence])])
        return self.index2Word[np.argmax(prediction[0])]

if __name__ == '__main__':
    def loadData(imagePaths, captions):
        '''Load images from given image paths and corresponding captions.
        70-30 Training-Test Split.'''

        images = []
        for path in imagePaths:
            img = cv2.imread(path)
            # Resize because VGG is trained for 224x224x3 images.
            img.resize((3, 224, 224))
            images.append(img)
        images = np.array(images)

        split = int(0.7 * len(images))

        return images[:split], captions[:split], images[split:], captions[split:]

    captions = ["START A dog is jumping over a bar END",
            "START A little boy is playing with toys END",
            "START A man is doing a backflip END",
            "START A man is playing a guitar END",
            "START A man is surfing on a wave END"]

    imagePaths = ["images/dog.png",
             "images/boy.png",
             "images/backflip.png",
             "images/guitar.png",
             "images/surf.png"]

    Xtr, Ytr, Xte, Yte = loadData(imagePaths, captions)

    imageCaptioner = ImageCaptioner(Xtr, Ytr)
    imageCaptioner.train()

    '\nPredicting Captions:'

    for i in range(len(Xte)):
        predicted = ''
        predictedCaption = 'START'
        while predicted is not 'END' or i < 5:
            predicted = imageCaptioner.predict(Xte[i], predictedCaption)
            print predicted
            predictedCaption += ' ' + str(predicted)
            i += 1

        print '\nTest Image [', i, ']:'
        print 'Actual Caption: ', Yte[i]
        print 'predicted Caption: ', predictedCaption
