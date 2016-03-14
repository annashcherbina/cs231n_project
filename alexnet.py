from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
#AlexNet with batch normalization in Keras
#input image is 224x224
def alexnet(): 
    model = Sequential()
    model.add(Convolution2D(96, 7, 7, subsample=(1,1),input_shape=(3,64,64),border_mode='valid'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=tuple([1,1])))

    model.add(Convolution2D(128, 7, 7, subsample=(1,1),border_mode='valid'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=tuple([1,1])))

    model.add(Convolution2D(192,3, 3, subsample=(1,1),border_mode='valid'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=tuple([1,1])))

    #model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    #model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dense(4096))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('softmax'))
    return model 
