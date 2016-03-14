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
    #pretrained_model=VGG_16('vgg16_weights.h5')
    pretrained_model=VGG_16('vgg16_weights_dropout_regularization_augmenteddataTintContrast.stupidsgd.hdf5')
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    pretrained_model.compile(optimizer=sgd, loss='categorical_crossentropy',trainLayersIndividually=0)    
    #do some training! 
    print "compilation finished, fitting model" 

    print "pretrained_model.trainLayersIndividually:"+str(pretrained_model.trainLayersIndividually) 
    if pretrained_model.trainLayersIndividually==1: 
        train_epochs=5 
    else: 
        train_epochs=5     

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
    
    #print "loaded data from pickle" 
    #OPTIONAL: save loaded/pre-processed data to a pickle to save time in the future
    
    #print "saving preprocessed data to hdf5 file" 
    '''
    f=h5py.File('imagenet.transpose.individually.hdf5','w')
    dset_xtrain=f.create_dataset("X_train",data=X_train)
    dset_ytrain=f.create_dataset("Y_train",data=Y_train) 
    dset_xvalid=f.create_dataset("X_valid",data=X_valid) 
    dset_yvalid=f.create_dataset("Y_valid",data=Y_valid) 
    dset_xtest=f.create_dataset("X_test",data=X_test) 
    f.flush() 
    f.close() 
    '''
    #print "done saving pre-processed data to hdf5 file!" 
    #pretrained_model = pretrained('pretrained_model.h5',False)
    #pretrained_model=pretrained_finetune('assignment3_weights_dropout_regularization_augmenteddataTintContrast.stupidsgd.hdf5',False) 
    history=pretrained_model.fit(X_train, Y_train, 128,train_epochs,validation_data=tuple([X_valid,Y_valid]),verbose=1,show_accuracy=True)
    pretrained_model.save_weights("vgg16_weights_dropout_regularization_augmenteddataTintContrast.stupidsgd.hdf5",overwrite=True) 
    class_predictions=pretrained_model.predict_classes(X_test) 
    np.savetxt('vgg16_class_predictions_dropout_regularization_augmenteddataTintContrast.stupidsgd.txt',class_predictions,fmt='%i',delimiter='\t') 
    train_scores=pretrained_evaluate(pretrained_model,X_train,Y_train)
    print "pretrained model training scores:"+str(train_scores) 
    valid_scores=pretrained_evaluate(pretrained_model,X_valid,Y_valid)
    print "pretrained validation scores:"+str(valid_scores)


    print "writing out the predictions file" 
    predictions=open('vgg16_class_predictions_dropout_regularization_augmenteddataTintContrast.stupidsgd.txt','r').read().split('\n') 
    while '' in predictions: 
        predictions.remove('') 
        
    wnids=open(labels,'r').read().split('\n') 
    while '' in wnids: 
        wnids.remove('') 

    cur_dir=test_dir+"images/"
    onlyfiles = [f for f in listdir(cur_dir) if isfile(join(cur_dir, f))]
    entries=10000
    outf=open('vgg16_class_predictions_dropout_regularization_augmenteddataTintContrast.formatted.stupidsgd.txt','w') 

    for i in range(entries): 
        image_name=onlyfiles[i] 
        predict_index=int(predictions[i])
        wnid1=wnids[predict_index]
        outf.write(image_name+'\t'+str(wnid1)+'\n')

if __name__=="__main__":
    main() 
