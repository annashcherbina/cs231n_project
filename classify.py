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

    data=h5py.File('imagenet.transpose.individually.augment.contrast.tint.hdf5','r')
    X_train=np.asarray(data['X_train']) 
    Y_train=np.asarray(data['Y_train']) 
    X_valid=np.asarray(data['X_valid']) 
    Y_valid=np.asarray(data['Y_valid']) 
    X_test=np.asarray(data['X_test']) 
    '''
    #print "loaded data from pickle" 
    #OPTIONAL: save loaded/pre-processed data to a pickle to save time in the future
    
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
    
    #assignment 3 convnet with pre-trained weights 
    #pretrained_model = pretrained('pretrained_model.h5')
    pretrained_model = pretrained('pretrained_model.h5',True)
    #pretrained_model=pretrained_finetune('assignment3_weights.hdf5') 
    #pretrained_model=pretrained_finetune('assignment3_weights.hdf5',freezeAndStack=True) 
    sgd = SGD(lr=1e-1)#, decay=1e-6, momentum=0.9, nesterov=True)
    #adagrad=Adagrad() 
    pretrained_model.compile(optimizer=sgd, loss='categorical_crossentropy',trainLayersIndividually=1)
    #do some training! 
    print "compilation finished, fitting model" 

    print "pretriained_model.trainLayersIndividually:"+str(pretrained_model.trainLayersIndividually) 
    if pretrained_model.trainLayersIndividually==1: 
        train_epochs=5 
    else: 
        train_epochs=17 
    
    history=pretrained_model.fit(X_train, Y_train, 128,train_epochs,verbose=1,show_accuracy=True)
    while pretrained_model.trainLayersIndividually==1: 
        pretrained_model.freezeAndStack() 
        history=pretrained_model.fit(X_train, Y_train, 128, train_epochs,verbose=1,show_accuracy=True)

    #pretrained_model.save_weights("assignment3_weights.hdf5",overwrite=True) 
    pretrained_model.save_weights("assignment3_freeze_and_stack_weights.hdf5",overwrite=True) 


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
    class_predictions=pretrained_model.predict_classes(X_test) 

    #save all the outputs! 
    #output=open('pretrained_results.pkl','w') 
    sys.setrecursionlimit(50000) 
    output=open('pretrained_results_freezeAndStack.pkl','wb') 
    #pickle.dump(history,output) 
    pickle.dump(train_scores,output) 
    pickle.dump(valid_scores,output) 
    pickle.dump(predictions,output) 
    pickle.dump(class_predictions,output) 
    output.close() 

    

if __name__=="__main__":
    main() 
