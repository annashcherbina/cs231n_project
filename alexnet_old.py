from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

def alexnet_train(X_train,Y_train,batch_size=32,nb_epoch=10):
    #AlexNet with batch normalization in Keras
    print "compiling alexnet model" 
    model = Sequential()
    model.add(Convolution2D(64, 11, 11, border_mode='valid',input_shape=(3,64,64)))
    #model.add(BatchNormalization((64,66,66)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(128, 7, 7, border_mode='valid'))
    #model.add(BatchNormalization((128,35,35)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(192, 3, 3, border_mode='valid'))
    #model.add(BatchNormalization((128,32,32)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    #model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    #model.add(BatchNormalization((128,28,28)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(12*12*256))#, 4096))#, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096))# 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(200))#, init='normal'))
    model.add(BatchNormalization(200))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print "done compiling alexnet model!"
    print "fitting alexnet model..." 
    model.fit(X_train, Y_train, batch_size, nb_epoch,verbose=1,show_accuracy=True)
    model.save_weights("alexnet_model_weights.hdf5",overwrite=True) 
    print "done fitting alexnet model!" 
    return model

def alexnet_evaluate(model,X,Y,batch_size=100):
    print "evaluating alexnet model!" 
    scores=model.evaluate(X,Y,batch_size,show_accuracy=True,verbose=1)
    print "alexnet model evaluation is complete" 
    return scores 
