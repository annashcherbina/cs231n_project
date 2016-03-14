from Params import * 
from load_data import *
from preprocess import *
from vgg_like_convnet import *
from alexnet import * 
from vgg16_keras import * 
from pretrained import * 
import h5py 

import keras
import theano
#for visualization:
from keras.utils.visualize_util import plot
from keras.optimizers import SGD, Adagrad, Adadelta 

import pickle
import numpy as np 
import sys

def main():
    #load data 
    #X_train,Y_train,X_valid,Y_valid,X_test=load_data(training_dir,valid_dir,test_dir,labels,sample)
    #preprocess data by mean subtraction and normalization 
    #X_train,X_valid,X_test=preprocess(X_train,X_valid,X_test)
    #del X_train
    #del X_test

    #or load pre-processed data from a previously saved hdf5 file:

    data=h5py.File('imagenet.hdf5','r')
    X_train=np.asarray(data['X_train']) 
    Y_train=np.asarray(data['Y_train']) 
    X_valid=np.asarray(data['X_valid']) 
    Y_valid=np.asarray(data['Y_valid']) 
    X_test=np.asarray(data['X_test']) 
    print(Y_valid.shape) 
    print(X_valid.shape) 
    #print "loaded data from pickle" 
    #OPTIONAL: save loaded/pre-processed data to a pickle to save time in the future
    '''
    print "saving preprocessed data to hdf5 file" 
    f=h5py.File('imagenet.hdf5','w')
    dset_xtrain=f.create_dataset("X_train",data=X_train)
    dset_ytrain=f.create_dataset("Y_train",data=Y_train) 
    dset_xvalid=f.create_dataset("X_valid",data=X_valid) 
    dset_yvalid=f.create_dataset("Y_valid",data=Y_valid) 
    dset_xtest=f.create_dataset("X_test",data=X_test) 
    f.flush() 
    f.close() 
    print "done saving pre-processed data to hdf5 file!" 
    '''
    #train a VGG-like convent

    vgg_model,history=vgg_train(X_train,Y_train)    
    train_scores=vgg_evaluate(vgg_model,X_train,Y_train)
    print "VGG-like net training scores:"+str(train_scores) 
    valid_scores=vgg_evaluate(vgg_model,X_valid,Y_valid)
    print "VGG-like net validation scores:"+str(valid_scores)
    #Visualize the pretty model
    plot(vgg_model,to_file="vgg_like_convnet.png") 
    predictions=vgg_model.predict(X_test,verbose=1) 
    class_predictions=vgg_model.predict_classes(X_test) 
    #save all the outputs! 
    sys.setrecursionlimit(50000) 
    output=open('vgg_like_results.pkl','w') 
    pickle.dump(history,output) 
    pickle.dump(train_scores,output) 
    pickle.dump(valid_scores,output) 
    pickle.dump(predictions,output) 
    pickle.dump(class_predictions,output) 
    output.close() 

    #train a Keras version of the ConvNet implemented in Assignment#2 in class
    #TODO

    #train AlexNet
    '''
    alexnet_model=alexnet_train(X_train,Y_train)    
    train_scores=alexnet_evaluate(alexnet_model,X_train,Y_train)
    print "AlexNet training scores:"+str(train_scores) 
    valid_scores=alexnet_evaluate(alexnet_model,X_valid,Y_valid)
    print "AlexNet validation scores:"+str(valid_scores)
    #Visualize the pretty model
    plot(alexnet_model,to_file="alexnet_like_convnet.png") 


    #VGG-16 with pretrained weights
    vgg16_model = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    vgg16_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print "compiled vgg16"
    train_scores=vgg16_evaluate(vgg16_model,X_train,Y_train)
    print "vgg16 training scores:"+str(train_scores) 
    valid_scores=vgg16_evaluate(vgg16_model,X_valid,Y_valid)
    print "vgg16 validation scores:"+str(valid_scores)
    #Visualize the pretty model
    plot(vgg16_model,to_file="vgg16_convnet.png") 

    
    #assignment 3 convnet with pre-trained weights 
    #pretrained_model = pretrained('pretrained_model.h5')
    pretrained_model=pretrained_finetune('assignment3_weights.hdf5') 
    sgd = SGD(lr=1e-1)#, decay=1e-6, momentum=0.9, nesterov=True)
    #adagrad=Adagrad() 
    pretrained_model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    #do some training! 
    print "compilation finished, fitting model" 
    history=pretrained_model.fit(X_train, Y_train, 128, 20,verbose=1,show_accuracy=True)
    pretrained_model.save_weights("assignment3_weights.hdf5",overwrite=True) 
    train_scores=pretrained_evaluate(pretrained_model,X_train,Y_train)
    print "pretrained model training scores:"+str(train_scores) 
    valid_scores=pretrained_evaluate(pretrained_model,X_valid,Y_valid)
    print "pretrained validation scores:"+str(valid_scores)
    #Visualize the pretty model
    plot(pretrained_model,to_file="pretrained_convnet.png") 
    #run the model on our test data 
    print "getting predictions:" 
    predictions=pretrained_model.predict(X_test,verbose=1) 
    print "getting class predictions:" 
    class_predictions=pretrained_model.predict(X_test) 
    #save all the outputs! 
    output=open('pretrained_results.pkl','wb') 
    pickle.dump(history,output) 
    pickle.dump(train_scores,output) 
    pickle.dump(valid_scores,output) 
    pickle.dump(predictions,output) 
    pickle.dump(class_predictions,output) 
    output.close() 
    '''
    

if __name__=="__main__":
    main() 
