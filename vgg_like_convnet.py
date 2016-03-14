#Example VGG-like convnet from keras tutorial
#http://keras.io/examples/
from keras.models import Sequential
from keras.regularizers import WeightRegularizer, ActivityRegularizer 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,Adadelta,Adagrad,Adam


def vgg_train(weights=None):
    print "Compiling VGG Model..." 
    model = Sequential()
    # input: 64x64 images with 3 channels -> (3, 64, 64) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3,64,64)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256,W_regularizer=WeightRegularizer(l1=1e-6,l2=1e-6)))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))

    model.add(Dense(200,W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))
    model.add(Activation('softmax'))
    if weights!=None: 
        model.load_weights(weights) 
    return model 
