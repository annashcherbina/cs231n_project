#Example VGG-like convnet from keras tutorial
#http://keras.io/examples/
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


def vgg_train(X_train,Y_train,batch_size=128,nb_epoch=100):
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
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(200))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print "done compiling vgg model!"
    print "fitting vgg model..." 
    model.fit(X_train, Y_train, batch_size, nb_epoch,verbose=1,show_accuracy=True)
    model.save_weights("vgg_model_weights.hdf5",overwrite=True) 
    print "done fitting vgg model!" 
    return model

def vgg_evaluate(model,X,Y,batch_size=100):
    print "evaluating vgg model!" 
    scores=model.evaluate(X,Y,batch_size,show_accuracy=True,verbose=1)
    print "vgg model evaluation is complete" 
    return scores 
