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
    
    data=h5py.File('imagenet.transpose.individually.hdf5','r')
    X_train=np.asarray(data['X_train']) 
    Y_train=np.asarray(data['Y_train']) 
    X_valid=np.asarray(data['X_valid']) 
    Y_valid=np.asarray(data['Y_valid']) 
    X_test=np.asarray(data['X_test']) 
    
    
    
    pretrained_model=pretrained_finetune('assignment3_weights_nodropout_noregularization_augmenteddata.hdf5',freezeAndStack=False) 
    sgd = SGD(lr=1e-1)#, decay=1e-6, momentum=0.9, nesterov=True)
    pretrained_model.compile(optimizer=sgd, loss='categorical_crossentropy',trainLayersIndividually=0)
    valid_scores=pretrained_evaluate(pretrained_model,X_valid,Y_valid)
    print "pretrained training scores:"+str(valid_scores)

    #Visualize the pretty model
    #plot(pretrained_model,to_file="pretrained_convnet.png") 

    #run the model on our test data 
    print "predicting on evaluation data:"
    #evaluate_predictions=pretrained_model.predict(X_valid,verbose=1) 
    #np.savetxt('valid.out', evaluate_predictions, fmt='%f')
    #print "predicting on test data:"
    #predictions=pretrained_model.predict(X_test,verbose=1) 
    #np.savetxt('test.raw.out',predictions,fmt='%f') 
    #print "getting class predictions on test data:" 
    #class_predictions=pretrained_model.predict_classes(X_train) 
    #np.savetxt('train.out',class_predictions,fmt='%i') 
    #print "validation data predictions:"+str(evaluate_predictions) 
    #print "test predictions:"+str(predictions) 
    #save all the outputs! 
    #sys.setrecursionlimit(50000) 
    #outputf=open('qc.pkl','w') 
    #output=open('pretrained_results_freezeAndStack.pkl','wb') 
    #pickle.dump(history,outputf) 
    #pickle.dump(train_scores,outputf) 
    #pickle.dump(valid_scores,outputf) 
    #pickle.dump(predictions,outputf) 
    #pickle.dump(class_predictions,outputf) 
    #outputf.close() 

    

if __name__=="__main__":
    main() 
